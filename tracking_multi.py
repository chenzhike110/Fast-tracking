import torch
import cv2
import argparse
import imutils
import time
import numpy as np

from imutils.video import VideoStream
from imutils.video import FPS
from PIL import Image

from yolo.yolo import YOLO
from siamfcpp.utils.crop import get_crop, get_subwindow_tracking
from siamfcpp.utils.bbox import cxywh2xywh, xywh2cxywh, xyxy2cxywh
from siamfcpp.utils.misc import imarray_to_tensor, tensor_to_numpy
from siamfcpp.model_build import build_model
from siamfcpp.tracking_utils import postprocess_box, postprocess_score, restrict_box, cvt_box_crop2frame
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

print(cv2.__version__)

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str, default="../SRTP/video/video5.mp4",
                help="path to input video file")
args = vars(ap.parse_args())

hyper_params = dict(
    total_stride=8,
    context_amount=0.5,
    test_lr=0.52,
    penalty_k=0.04,
    window_influence=0.21,
    windowing="cosine",
    z_size=127,
    x_size=303,
    num_conv3x3=3,
    min_w=10,
    min_h=10,
    phase_init="feature",
    phase_track="track",
)

# multiprocess
dataqueues = []
resultqueues = []
process = []
track_object = {}
process_num = 4
scale_size = 1
knn_updated = False

def get_point(event, x, y, flags, param):
    global knn_updated, track_object
    if event == cv2.EVENT_LBUTTONDOWN:
        if knn_updated == True:
            return 
        mousex = x
        mousey = y
        find = False
        for j in track_object.keys():
            if x>track_object[j][0][0]and x<track_object[j][0][0]+track_object[j][0][2]and y>track_object[j][0][1] and y<track_object[j][0][1]+track_object[j][0][3]:
                find = True
                index = j
                break
        if find == False:
            return
        key = cv2.waitKey(0) & 0xFF
        while key == 255:
            key = cv2.waitKey(0) & 0xFF
        if key == ord(" "):
            track_object[j][1]= 2
        elif key == 13: #enter键
            track_object[j][1]= 3
        else:
            track_object[j][1]= 4

def command_process(commandqueue):
    hostname = socket.gethostname() #真机
    hostip_real = socket.gethostbyname(hostname+".local") #真机
    print('host ip', hostip_real)
    port = 6666  # Arbitrary non-privileged port
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1)
    # s.bind((hostip_real[3], port)) #真机
    s.bind((hostip_real, port)) #模拟1
    s.listen(4)

    while True:
        try:
            command = commandqueue.get()
        except:
            continue
        else:
            try: 
                conn, addr = s.accept()
                conn.settimeout(1)
                # cmd = "0,0,0,0\n"
                data = conn.recv(1024)
                if not data:
                    print('no data')
                data = data.decode().split(',')
                # x速度，y速度，z速度，纬度，经度，高度，电池是否过低，是否飞行
                print(data)

                # 发送 eg：0,0,0,0
                # number1：-1~1：-1表示最大速度下降；1表示最大速度上升
                # number2：-1~1：-1表示最大速度向左yaw；1表示最大速度向右yaw
                # number3：-1~1：-1表示最大速度向前；1表示最大速度向后
                # number4：-1~1：-1表示最大速度向左；1表示最大速度向右
                # cmd = input()
                cmd = ','.join(map(str, command))+"\n"
                conn.sendall(cmd.encode())  # 数据发送
                # print(cmd.encode())
                conn.close()
            except:
                print('timeout')
            else:
                print("send: ",cmd)


def single_process(index, image, dataqueue, resultqueue):
    device = torch.device("cuda")
    model = build_model("siamfcpp/models/siamfcpp-tinyconv-vot.pkl").to(device)
    # parameters
    avg_chans = np.mean(image, axis=(0, 1))
    z_size = hyper_params['z_size']
    x_size = hyper_params['x_size']
    context_amount = hyper_params['context_amount']
    phase = hyper_params['phase_init']
    phase_track = hyper_params['phase_track']
    score_size = (hyper_params['x_size'] -hyper_params['z_size']) // hyper_params['total_stride'] + 1 - hyper_params['num_conv3x3'] * 2
    window = np.outer(np.hanning(score_size), np.hanning(score_size))
    window = window.reshape(-1)
    im_h, im_w = image.shape[0], image.shape[1]
    total_num = 0
    # property
    # [state, features, lost]
    tracking_index = {}
    print("This is process",index)

    def init(state, im_x, total_num):
        print("init start")
        for i in range(len(state)):
            tracking_index[index*100+total_num+i] = [state[i]]
            im_z_crop, _ = get_crop(im_x, state[i][:2], state[i][2:4], z_size, avg_chans=avg_chans, context_amount=context_amount, func_get_subwindow=get_subwindow_tracking)
            array = torch.from_numpy(np.ascontiguousarray(im_z_crop.transpose(2, 0, 1)[np.newaxis, ...], np.float32)).to(device)
            with torch.no_grad():
                tracking_index[index*100+total_num+i].append(model(array,phase=phase))
            tracking_index[index*100+total_num+i].append(0)
    
    def delete_node(j):
        try: 
            del tracking_index[j]
        except Exception as error:
            print("delete error",error)
    
    while True:
        try: 
            im_x, state, delete = dataqueue.get(timeout=1)
        except Exception as error:
            print(error)
            continue
        else:
            if len(state) > 0:
                init(state, im_x, total_num)
                print("init success")
                total_num += len(state)
                continue
            if len(delete) > 0:
                delete_list = []
                for i in delete:
                    if i in tracking_index:
                        print("delete",i)
                        delete_node(i)
            
            result = []
            for i in tracking_index.keys():
                im_x_crop, scale_x = get_crop(im_x, tracking_index[i][0][:2], tracking_index[i][0][2:4], z_size, x_size=x_size, avg_chans=avg_chans,context_amount=context_amount, func_get_subwindow=get_subwindow_tracking)
                array = torch.from_numpy(np.ascontiguousarray(im_x_crop.transpose(2, 0, 1)[np.newaxis, ...], np.float32)).to(device)
                with torch.no_grad():
                    score, box, cls, ctr, *args = model(array, *tracking_index[i][1], phase=phase_track)
                
                box = tensor_to_numpy(box[0])
                score = tensor_to_numpy(score[0])[:, 0]
                cls = tensor_to_numpy(cls[0])
                ctr = tensor_to_numpy(ctr[0])
                box_wh = xyxy2cxywh(box)

                # lost goal
                if score.max()<0.2:
                    tracking_index[i][2] += 1
                    result.append([cxywh2xywh(np.concatenate([tracking_index[i][0][:2], tracking_index[i][0][2:4]],axis=-1)), tracking_index[i][2]])
                    continue
                elif tracking_index[i][2] > 0:
                    tracking_index[i][2] -= 1
                best_pscore_id, pscore, penalty = postprocess_score(score, box_wh, tracking_index[i][0][2:4], scale_x, 0.4, window, 0.5)
                # box post-processing
                new_target_pos, new_target_sz = postprocess_box(best_pscore_id, score, box_wh, tracking_index[i][0][:2], tracking_index[i][0][2:4], scale_x, x_size, penalty, 0.8)
                new_target_pos, new_target_sz = restrict_box(new_target_pos, new_target_sz, im_w, im_h, 10, 10)

                # save underlying state
                tracking_index[i][0] = np.append(new_target_pos, new_target_sz)

                # return rect format
                track_rect = cxywh2xywh(np.concatenate([new_target_pos, new_target_sz],axis=-1))
                result.append([track_rect,tracking_index[i][2]])
            
            delete_list = []
            for i in tracking_index.keys():
                if tracking_index[i][2] > 10:
                    delete_list.append(i)
            
            for i in delete_list:
                delete_node(i)
            
            resultqueue.put([result, list(tracking_index.keys())])

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('forkserver', force=True)
    if not args.get("video", False):
        print("[INFO] starting video stream...")
        vs = VideoStream(src=0).start()
    else:
        vs = cv2.VideoCapture(args["video"])

    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame
    frame = imutils.resize(frame, height=frame.shape[0]//scale_size, width=frame.shape[1]//scale_size)
    initBB = None

    cv2.namedWindow('result')
    cv2.setMouseCallback('result',get_point)
    # Detector and Tracker initial
    yolo = YOLO()

    # start process
    for i in range(process_num):
        dataqueues.append(torch.multiprocessing.Queue())
        resultqueues.append(torch.multiprocessing.Queue())
        process.append(torch.multiprocessing.Process(target=single_process, args=(i, frame, dataqueues[-1], resultqueues[-1],)))
        process[-1].start()

    commandqueue = torch.multiprocessing.Queue()
    controlProcess = torch.multiprocessing.Process(target=command_process, args=(commandqueue,))
    controlProcess.start()

    while True:
        frame = vs.read()
        frame = frame[1] if args.get("video", False) else frame
        frame = imutils.resize(frame, height=frame.shape[0]//scale_size, width=frame.shape[1]//scale_size)
        if frame is None:
            break
        # update the FPS counter
        
        if initBB is not None:
            for i in range(len(dataqueues)):
                dataqueues[i].put((frame, [], []))
            for i in range(len(resultqueues)):
                try:
                    result, indexes = resultqueues[i].get()
                except RuntimeError:
                    print("lost")
                except Exception as error:
                    print("empty")
                else:
                    for index, j in enumerate(indexes):
                        try:
                            if track_object[j][1] > 10:
                                del track_object[j]
                            else:
                                track_object[j][0]=result[index][0]
                        except KeyError:
                            print("main deleted but process not")
                            continue
            for i in track_object.keys():
                (x, y, w, h) = [int(v) for v in track_object[i][0]]
                txt = int(track_object[i][1])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255 - txt*50, txt*50), 2)
                cv2.putText(frame, "{}".format(track_object[i][1]), (x + w//2, y + h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255 - txt*50, txt*50), 1)
            fps.update()
            fps.stop()
        else:
            starttime = time.time()
            img_new = Image.fromarray(np.uint8(frame))
            initBB = yolo.detect_image_without_draw(img_new)
            starttime = time.time()
            for i in range(len(dataqueues)):
                temp = initBB[int(len(initBB)/len(dataqueues)*i):int(len(initBB)/len(dataqueues)*(i+1))]
                for j in range(len(temp)):
                    track_object[i*100+j] = [cxywh2xywh(temp[j][0:4]),temp[j][-1]]
                dataqueues[i].put((frame, temp, [])) 
            fps = FPS().start()

        cv2.imshow("result", frame)
        cv2.waitKey(1)