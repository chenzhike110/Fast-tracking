import torch
import cv2
import argparse
import imutils
import time
import numpy as np
import socket

from imutils.video import VideoStream
from imutils.video import FPS
from PIL import Image

from yolo.yolo import YOLO
from siamfcpp.multi_tracker import Multi_Tracker
from siamfcpp.utils.bbox import cxywh2xywh, xywh2cxywh, xyxy2cxywh
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
from ORBmin import KNNClassifier,get_data_from_video,mini_img,init_get_video,len_all


warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

print(cv2.__version__)

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str, default="../SRTP/video/video5.mp4",
                help="path to input video file")
args = vars(ap.parse_args())

# multiprocess
dataqueues = []
resultqueues = []
process = []
track_object = {}
process_num = 3
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
        key = cv2.waitKey(1) & 0xFF
        while key == 255:
            key = cv2.waitKey(0) & 0xFF
        if key == ord("a"):
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
                print('timeout', ','.join(map(str, command))+"\n")
            else:
                print("send: ",cmd)

def Control(x,y,centerX,centerY):
    # print(x,y,centerX,centerY)
    commandx = 2*(centerX-x)/(2*centerX)
    commandy = 2*(centerY-y)/(2*centerY)
    command = np.array([0,0,-commandy,commandx])
    np.clip(command,-1,1)
    return command

if __name__ == "__main__":
    KNN=None
    videoname=args['video'].split('/')[-1]
    model_path='knn_class'
    num_of_photo=25
    classes_name=['plyaer','ball','team1','team2','judger']#0,1,2,3,4
    padding=10
    update_data=False
    # 单个bao
    
    knn_updated=init_get_video(classname=classes_name[2:],video_name=videoname,num_of_photo=num_of_photo, path=model_path,update_data=update_data)
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
    torch.multiprocessing.set_start_method(method='spawn')
    # start process
    for i in range(process_num):
        dataqueues.append(torch.multiprocessing.Queue())
        resultqueues.append(torch.multiprocessing.Queue())
        worker = Multi_Tracker(i, frame, dataqueues[-1], resultqueues[-1])
        worker.start()

    commandqueue = torch.multiprocessing.Queue()
    controlProcess = torch.multiprocessing.Process(target=command_process, args=(commandqueue,))
    controlProcess.start()

    yolo = YOLO()

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
                    # print("Main get",indexes)
                    for index, j in enumerate(indexes):
                        try:
                            if result[index][1] > 10:
                                print("delete",j)
                                del track_object[j]
                            else:
                                track_object[j][0]=result[index][0]
                                if track_object[j][1]==2 or track_object[j][1]==3 or track_object[j][1]==4 and knn_updated==False :
                                    if KNN==None:
                                        get_data_from_video(frame=frame, box=track_object[j][0], classname=classes_name[track_object[j][1]], padding=padding, video_name=videoname,path=model_path,num_of_photo=num_of_photo)

                                        photoum=len_all(path=model_path, videoname=videoname,classes_name=classes_name[2:])

                                        knn_updated=True
                                        for i in photoum:
                                            if i<num_of_photo:
                                                knn_updated=False
                                                break
                                elif (track_object[j][1]==0) and KNN!=None:
                                    track_object[j][1],mat,box=KNN.prediction(box=track_object[j][0],frame=frame,video_name=videoname,classes_name=classes_name,
                                    padding=padding,
                                    save_img_recode=False,k=5)
                        except KeyError:
                            print("main deleted but process not", j)
                            continue
            if knn_updated:
                print('init KNNClassifier')
                KNN=KNNClassifier(video_name=videoname,modelpath=model_path)
                knn_updated=False

            for i in track_object.keys():
                (x, y, w, h) = [int(v) for v in track_object[i][0]]

                txt = int(track_object[i][1])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255 - txt*50, txt*50), 2)
                cv2.putText(frame, "{},{}".format(track_object[i][1],i), (x + w//2, y + h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255 - txt*50, txt*50), 1)
            fps.update()
            fps.stop()
            # print(fps.fps())
            
        else:
            img_new = Image.fromarray(np.uint8(frame))
            initBB = yolo.detect_image_without_draw(img_new)
            for i in range(len(dataqueues)):
                temp = initBB[int(len(initBB)/len(dataqueues)*i):int(len(initBB)/len(dataqueues)*(i+1))]
                for j in range(len(temp)):
                    track_object[i*100+j] = [cxywh2xywh(temp[j][0:4]),temp[j][-1]]
                dataqueues[i].put((frame, temp, [])) 
            fps = FPS().start()

        cv2.imshow("result", frame)
        cv2.waitKey(1)