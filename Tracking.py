import torch
import cv2
import numpy as np
import argparse
import imutils
import time
from imutils.video import VideoStream
from imutils.video import FPS
import socket

from Videoq import VideoCapture

from siamfcpp.model_build import build_model
from siamfcpp.Tracker import SiamFCppTracker

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
    upcommand = 10

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
                # number3：-1~1：1表示最大速度向前；-1表示最大速度向后
                # number4：-1~1：-1表示最大速度向左；1表示最大速度向右
                # cmd = input()
                cmd = ','.join(map(str, command))+"\n"
                # if upcommand > 0:
                #     cmd = '1,0,0,0'
                #     upcommand = upcommand - 1
                conn.sendall(cmd.encode())  # 数据发送
                # print(cmd.encode())
                conn.close()
            except:
                print('timeout: ', ','.join(map(str, command))+"\n")
            else:
                print("send: ",cmd)

def Control(x,y,centerX,centerY):
    # print(x,y,centerX,centerY)
    commandx = 1.5*(centerX-x)/(centerX)
    commandy = 1.5*(centerY-y)/(centerY)
    command = np.array([0,0,commandy,-commandx])
    np.clip(command,-1,1)
    return command
    
Model = build_model("siamfcpp/models/siamfcpp-tinyconv-vot.pkl")
tracker = SiamFCppTracker()
tracker.set_model(Model)
tracker.to_device(torch.device("cuda"))
commandqueue = torch.multiprocessing.Queue()
controlProcess = torch.multiprocessing.Process(target=command_process, args=(commandqueue,))
controlProcess.start()

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str, default="rtmp://127.0.0.1:9999/live/test",
                help="path to input video file")
args = vars(ap.parse_args())

initBB = None
# if a video path was not supplied, grab the reference to the web cam
if not args.get("video", False):
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
# otherwise, grab a reference to the video file
else:
    vs = VideoCapture(args["video"])
# initialize the FPS throughput estimator
fps = None
show = True
# loop over frames from the video stream
while True:
    # grab the current frame, then handle if we are using a
    # VideoStream or VideoCapture object
    frame = vs.read()
    # frame = frame[1] if args.get("video", False) else frame
    # check to see if we have reached the end of the stream
    if frame is None:
        break
    # resize the frame (so we can process it faster) and grab the
    # frame dimensions
    # (H, W) = frame.shape[:2]
    # frame = cv2.resize(frame, (W//4, H//4), interpolation=cv2.INTER_LINEAR)
    # (H, W) = frame.shape[:2]
    # frame = imutils.resize(frame, width=1024)
    # print(frame.shape)
    # check to see if we are currently tracking an object
    if initBB is not None:
        # grab the new bounding box coordinates of the object
        #(success, box) = pipeline.update(frame)
        box,fail = tracker.update(frame)
        success=not fail
        # check to see if the tracking was a success
        if success and show:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h),
                          (0, 255, 0), 2)
        # update the FPS counter
        fps.update()
        fps.stop()
        # print(fps.fps())
        # initialize the set of information we'll be displaying on
        # the frame
        if show: 
            # frame = imutils.resize(frame, width=1024)
            (H, W) = frame.shape[:2]
            info = [
                ("Tracker", "siamfcpp"),
                ("Success", "Yes" if success else "No"),
                ("FPS", "{:.2f}".format(fps.fps())),
            ]
            # loop over the info tuples and draw them on our frame
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
        command = Control(x+w/2,y+h/2,W/2,H/2)
        commandqueue.put(command)
    else:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            # select the bounding box of the object we want to track (make
            # sure you press ENTER or SPACE after selecting the ROI)
            initBB = cv2.selectROI("Frame", frame, fromCenter=False,
                                showCrosshair=True)
            # start OpenCV object tracker using the supplied bounding box
            # coordinates, then start the FPS throughput estimator as well
            tracker.init(frame, initBB)
            fps = FPS().start()

# if we are using a webcam, release the pointer
if not args.get("video", False):
    vs.stop()
# otherwise, release the file pointer
else:
    vs.release()
# close all windows
cv2.destroyAllWindows()