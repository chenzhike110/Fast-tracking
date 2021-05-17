#-------------------------------------#
#   调用摄像头或者视频进行检测
#   调用摄像头直接运行即可
#   调用视频可以将cv2.VideoCapture()指定路径
#   视频的保存并不难，可以百度一下看看
#-------------------------------------#
import time

import cv2
import numpy as np
from PIL import Image

from yolo import YOLO
from ball_touch import ball_state
from edge import offside_dectet
from kmeans_for_anchors import cas_iou


yolo = YOLO()
#-------------------------------------#
#   调用摄像头
#   capture=cv2.VideoCapture("1.mp4")
#-------------------------------------#
#frame = cv2.imread("../1.jpg")
capture=cv2.VideoCapture("/home/jiangcx/桌面/足球视频/video5.mp4")
#frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
# 转变成Image
#frame = Image.fromarray(np.uint8(frame))
#frame = np.array(yolo.detect_image(frame))
#frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
#frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#cv2.imshow("video",frame)
#cv2.waitKey(0)

fps = 0.0



while(True):
    t1 = time.time()
    # 读取某一帧
    ref,frame=capture.read()
    frame = cv2.resize(frame,(1920,1080))
    #cut = int(((frame.shape)[1] - (frame.shape)[0])/2)
    #frame = frame[:,cut:-cut,:]
    # 格式转变，BGRtoRGB
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    # 转变成Image
    frame = Image.fromarray(np.uint8(frame))
    # 进行检测
    frame,boxes,top_label = yolo.detect_image(frame)
    frame = np.array(frame)

   # state = 0:检测到大于一个球;1:触球;2:有球但未触球;3:球出边界所以没识别到;4:还未第一次检测到球;5:球在场内但没识别到
    state,touch_person = ball_state(frame,boxes, top_label)
    if state==1:
        # （图像，向哪个方向进攻(left & right)，进攻球员x，进攻球员y，防守球员x，防守球员y
        has_line, has_offside = offside_dectet(frame,'up', int(boxes[0,0]), int(boxes[0,1]), int(boxes[1,0]), int(boxes[1,1]))

    # RGBtoBGR满足opencv显示格式
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    fps  = ( fps + (1./(time.time()-t1)) ) / 2
    print("fps= %.2f"%(fps))
    frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("video",frame)

    c= cv2.waitKey(1) & 0xff 
    if c==27:
        capture.release()
        break
