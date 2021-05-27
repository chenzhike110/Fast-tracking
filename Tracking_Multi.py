import torch
import cv2
import argparse
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
from cosin import KNNClassifier,get_data_from_video,mini_img,init_get_video,len_all
from yolo.utils.utils import bbox_iou 
from easy_ball_track import ball_track, xywh_iou, get_distance
from edge import offside_dectet,draw_offside_line
# from Videoq import VideoCapture

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

print(cv2.__version__)

# def get_new(image):
#     # img_origin = image.copy()
#     # image = cv2.resize(image,(math.ceil(image.shape[1]/2),math.ceil(image.shape[0]/2)))
#     # has_offside = 0
#     th = 30  # 边缘检测后大于th的才算边界

#     gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
#     # gray_origin = cv2.cvtColor(img_origin, cv2.COLOR_BGRA2GRAY)
#     gray = cv2.GaussianBlur(gray, (5, 5), 0)

#     x = cv2.Sobel(gray, cv2.CV_16S, 1, 0)  # x方向梯度
#     y = cv2.Sobel(gray, cv2.CV_16S, 0, 1)  # y方向梯度
#     absX = cv2.convertScaleAbs(x)  # 转回uint8
#     absY = cv2.convertScaleAbs(y)
#     edges = cv2.addWeighted(absX, 0.5, absY, 0.5, 0) # 各0.5的权重将两个梯度叠加
#     edges = edges[:,:,np.newaxis]
#     image = np.concatenate((image, edges),axis=-1)
#     return image

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str, default="./video/auto1_2.mp4",
                help="path to input video file")
args = vars(ap.parse_args())

# multiprocess
dataqueues = []
resultqueues = []
tracking_number = []
process = []
track_object = {} # 0 xywh, 1 cls, 2 offside, 3 touchball
process_num = 2
scale_size = 1
knn_updated = False
direction = None
bias_keeper = 0
kicker = -1
teamcolor = [(0,255,0),(0,0,0),(9,161,255),(73,255,9),(0,0,255)]
# origin_direction = ["down","up"]
origin_direction = ["up","down"]

# def exchange_cls(cls_ball,defense):
#     temp_cls = cls_ball
#     cls_ball = defense
#     defense = temp_cls
#     return cls_ball,defense


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

# @number.jit()
def check_box(initBB):
    tracking_xy = np.array([value[0] for value in track_object.values()])
    if len(tracking_xy) == 0:
        return initBB
    add_box = []
    for i in range(len(initBB)):
        boundingbox = cxywh2xywh(initBB[i][0:4])
        iou = bbox_iou(torch.Tensor([boundingbox]), torch.Tensor(tracking_xy), False)
        if (iou > 0.1).any():
            continue
        boundingbox = xywh2cxywh(boundingbox)
        boundingbox = np.append(boundingbox,0)
        add_box.append(boundingbox)
    return add_box

if __name__ == "__main__":
    KNN=None
    videoname=args['video'].split('/')[-1]
    model_path='knn_class'
    num_of_photo=25
    classes_name=['player','ball','team1','team2','judger']#0,1,2,3,4
    padding=0
    update_data=False
    ball_vector=np.array([0,0])# 球向量
    ball_touch_frame=-1
    
    knn_updated=init_get_video(classname=classes_name[2:],video_name=videoname,num_of_photo=num_of_photo, path=model_path,update_data=update_data)
    if not args.get("video", False):
        print("[INFO] starting video stream...")
        vs = VideoStream(src=0).start()
    else:
        vs = cv2.VideoCapture(args["video"])

    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame
    # frame = imutils.resize(frame, height=frame.shape[0]//scale_size, width=frame.shape[1]//scale_size)
    # frame = imutils.resize(frame, width=1920, height=1080)
    frame = cv2.resize(frame, (1920,1080))
    initBB = None
    fps = None

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    ffps = vs.get(cv2.CAP_PROP_FPS)
    #size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    #frame=frameflow.frame
    size=frame.shape[:2]
    out = cv2.VideoWriter('./camera_test12.mp4', fourcc, ffps, (size[1],size[0]))

    cv2.namedWindow('result')
    cv2.setMouseCallback('result',get_point)
    # Detector and Tracker initial
    torch.multiprocessing.set_start_method(method='spawn',force=True)
    # start process
    for i in range(process_num):
        tracking_number.append(0)
        dataqueues.append(torch.multiprocessing.Queue())
        resultqueues.append(torch.multiprocessing.Queue())
        worker = Multi_Tracker(i, frame, dataqueues[-1], resultqueues[-1])
        worker.start()

    # commandqueue = torch.multiprocessing.Queue()
    # controlProcess = torch.multiprocessing.Process(target=command_process, args=(commandqueue,))
    # controlProcess.start()

    balldatequeue=torch.multiprocessing.Queue()
    ballresultqueue=torch.multiprocessing.Queue()
    ballTrack=torch.multiprocessing.Process(target=ball_track,args=(balldatequeue,ballresultqueue))
    ballTrack.start()

    yolo = YOLO()
    framecount = -1

    while True:
        # for i in range(1000):
        #     frame=vs.read()
        frame = vs.read()
        frame = frame[1] if args.get("video", False) else frame
        if frame is None:
            break
        #frame = imutils.resize(frame, height=frame.shape[0]//scale_size, width=frame.shape[1]//scale_size)
        frame = cv2.resize(frame, (1920,1080))
        # frame = imutils.resize(frame, width=1920, height=1080)       

        origin_frame = frame.copy()
        framecount += 1

        balldatequeue.put((frame,None))
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
            # if KNN is not None:
            #     for i in track_object.keys():
            #         track_object[i][1], mat, box = KNN.prediction(box=track_object[i][0], frame=frame,video_name=videoname, classes_name=classes_name,
            #                                                       padding=padding,
            #                                                       save_img_recode=False, k=5)
            
            for i in track_object.keys():
                (x, y, w, h) = [int(v) for v in track_object[i][0]]

                txt = int(track_object[i][1])
                cv2.rectangle(frame, (x, y), (x + w, y + h), teamcolor[txt], 2)
                if track_object[i][2] and track_object[i][3]:
                    cv2.putText(frame, "OFFSIDE !!!", (x + w//2, y + h), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0,0,255), 10)
        
        try:
            pred,touch=ballresultqueue.get()
            print("----------------------------------------------------------------",touch)
            x,y,w,h=pred

        except Exception as Err:
            pass
        else:
            # 更新球向量
            if framecount!=0:
                v_2 = np.array([x+w//2,y+h//2])
                ball_vector = v_2-v_1
            else:
                v_1 = np.array([x+w//2,y+h//2])
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 3)
            if touch:# 如果判断存在触球
                cls_ball = 0
                paddling_iou = 10
                ball_touch_frame=3
                dis_near_player = 50
                kick_ball = False
                tracking_xy = np.array([value[0] for value in track_object.values()])
                distance = np.array(get_distance(torch.Tensor([pred]),torch.Tensor(tracking_xy)))
                iou = xywh_iou(torch.Tensor([[pred[0]-paddling_iou,pred[1]-paddling_iou,pred[2]+2*paddling_iou,pred[3]+2*paddling_iou]]), torch.Tensor(tracking_xy))
                if len(distance[distance<dis_near_player**2])>1: # situation that is hard to distinguish the ownership of ball
                    if kicker > 0:
                        track_object[kicker][3] = False
                    kicker = -1
                    kick_ball = True
                elif (iou > 0).any():
                    index = int(np.argmax(iou))
                    if kicker > 0:
                        track_object[kicker][3] = False
                    kicker = [touch_player for touch_player, value in track_object.items() if (tracking_xy[index] == value[0]).all()][0]
                    track_object[kicker][3] = True
                    if track_object[kicker][1] == 2:
                        direction = origin_direction[0]
                    elif track_object[kicker][1] == 3:
                        direction = origin_direction[1]
                    kick_ball = True
                if direction=="up":
                    ofplayers = [value[0] for value in track_object.values() if value[1] == origin_direction.index(direction)+2]
                    dfplayer = [value[0] for value in track_object.values() if value[1] == 5-(origin_direction.index(direction)+2)]
                    dfplayer = sorted(dfplayer,key=lambda s: s[1],reverse = False)
                elif direction=="down":
                    ofplayers = [value[0] for value in track_object.values() if value[1] == origin_direction.index(direction)+2]
                    dfplayer = [value[0] for value in track_object.values() if value[1] == 5-(origin_direction.index(direction)+2)]
                    dfplayer = sorted(dfplayer,key=lambda s: s[1],reverse = True)
                if direction!=None and kick_ball:
                    k,has_line,has_offside = offside_dectet(origin_frame,direction,ofplayers,dfplayer[bias_keeper])
                    if has_line: 
                        frame = draw_offside_line(frame,direction,dfplayer[bias_keeper],k)
                        for id_, value in enumerate(ofplayers):
                            for index, player in track_object.items():
                                if index == kicker:
                                    break
                                if (player[0] == value).all():
                                    track_object[index][2] = has_offside[id_]
                                    break
               
            if ball_touch_frame!=0:
                ball_touch_frame-=1
            elif ball_touch_frame==0:
                #判断触球
                # 人和球的
                pass
            if fps != None:
                fps.stop()
                fps.update()
            cv2.putText(frame, "FPS: {:.2f}".format(fps.fps()), (10,frame.shape[0]-200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
        if framecount % 100 == 0:
            # img_new = get_new(frame)
            img_new = Image.fromarray(np.uint8(origin_frame))
            initBB = yolo.detect_image_without_draw(img_new)
            initBB = mini_img(frame,initBB)
            initBB = check_box(initBB)
            for i in range(len(dataqueues)):
                temp = initBB[int(len(initBB)/len(dataqueues)*i):int(len(initBB)/len(dataqueues)*(i+1))]
                for j in range(len(temp)):
                    track_object[i*100+tracking_number[i]] = [temp[j][0:4],temp[j][-1],0,False,False]
                    tracking_number[i] += 1
                dataqueues[i].put((origin_frame, temp, []))  
            if fps == None:
                fps = FPS().start()

        cv2.imshow("result", frame)
        cv2.waitKey(1)
        out.write(frame)
    out.release()
