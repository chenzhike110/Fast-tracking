import math

import cv2
import cv2 as cv
import numpy as np
import time
from tqdm import tqdm
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity
from .circle import circle_dectet



def inplayer(box_ball,box_player):
    #x,y,w,h
    #x,y,w,h
    cx=box_ball[0]+box_ball[2]//2
    cy=box_ball[1]+box_ball[3]//2
    if (box_player[0]<cx) and (box_player[1]<cy) and (box_player[0]+box_player[2])>cx and (box_player[1]+box_player[3])>cy:
        return True
    else:
        return False

def ball_score(gray_test,test,ground_truth,tracking_object,up=False):
    global adaptive
    adaptive = 0
    #contours_person, hier = cv.findContours(gray_test, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours_person, hier = cv.findContours(gray_test, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    w_h = [] #长宽比
    c_s = [] #周长面积比
    ssim = [] #相似度
    d_target = [] #目标区域
    score = [] #得分

    # detect the circle
    has_circle, circles = circle_dectet(test)
    c_bound = 30
    #for circle in circles:
    #    cv.circle(test, (circle[0], circle[1]), c_bound, (255, 0, 0), 2)
    circles = np.array(circles)
    x_c = circles[:,0]
    y_c = circles[:,1]

    # xywh = list(map(cv.boundingRect,contours_person))
    # xywh = np.array(list(map(np.array,xywh)))
    # wh = np.min(temp[:,0]/temp[:,1],temp[:,1]/temp[:,0])
    # print(temp.shape)
    #for d in tqdm(contours_person):
    if up==True:
        contours_person=tqdm(contours_person)

    for d in contours_person:
        if d.shape[0]<10:
            continue
        x, y, w, h = cv.boundingRect(d)     # 获取矩形框边界坐标
        #计算长宽比
        if (w > h):
            wh = 1 / (w / h)
        else:
            wh = 1 / (h / w)
        # 计算矩形框的面积
        area_obj = cv.contourArea(d)
        perimeter = cv.arcLength(d, True)  # 周长
        cs = 4*math.pi*area_obj/(perimeter*perimeter)

        merge = 10 #剔除边界的阈值
        # 限制了识别区域的大小、长宽比、面积周长比、距离边缘距离等【可调整】
        if np.max([5,10-adaptive]) <= area_obj <np.min([800,100+adaptive]) and np.max([5,10-adaptive])<w*h<np.min([800,100+adaptive]) and cs>0.5 and wh>0.5 and  x > merge and y > merge and abs(x - test.shape[1])>merge and abs(y - test.shape[0])>merge and (has_circle == 0 or ((np.abs(x_c-x)>c_bound).all() and (np.abs(x_c-x)>c_bound).all())):
            
            if tracking_object!=None:#key:{[[x,y,w,h]int]}
                flag=0
                for key,value in tracking_object.items():
                    if value[1]!=1:
                        box=value[0]
                    else:
                        continue
                    if inplayer((x,y,w,h),box):
                        flag=1
                        break
                if flag:
                    continue
            #计算相似性
            scale = 2 #可调，感受野增量
            temp = test[y - scale:y + h + scale, x - scale:x + w + scale, :]
            temp = cv.resize(temp, (ground_truth.shape[1], ground_truth.shape[0]))

            sim = structural_similarity(ground_truth, temp, multichannel=True)
            # sim = 0

            if (sim>0.5):
                d_target.append(d) #待选区块
                w_h.append(wh)
                c_s.append(cs)
                ssim.append(sim)
    print('wh',w_h,'cs',c_s,'s',ssim)
    score = np.array(w_h)+np.array(c_s)+np.array(ssim)
    return d_target , score

def dective_by_background(test,ground_truth,tracking_object):

    th = 100

    gray_test = cv.cvtColor(test, cv.COLOR_BGRA2GRAY)
    gray_test = cv.GaussianBlur(gray_test, (5, 5), 0)

    x = cv.Sobel(gray_test, cv.CV_16S, 1, 0)  # x方向梯度
    y = cv.Sobel(gray_test, cv.CV_16S, 0, 1)  # y方向梯度
    absX = cv.convertScaleAbs(x)  # 转回uint8
    absY = cv.convertScaleAbs(y)
    edges = cv.addWeighted(absX, 0.5, absY, 0.5, 0)  # 各0.5的权重将两个梯度叠加

    dst, edges = cv.threshold(edges, th, 255, cv.THRESH_BINARY)  # 大于th的赋值255（白色）
    edges = cv2.dilate(edges,kernel=np.ones((3, 3), np.uint8),iterations=1)
    edges = cv2.erode(edges, kernel=np.ones((3, 3), np.uint8), iterations=1)
    hsv = cv.cvtColor(test, cv.COLOR_BGR2HSV)

    lower_white = np.array([0, 0, 130])
    upper_white = np.array([180, 80, 255])
    mask = cv.inRange(hsv, lower_white, upper_white)
    edges = cv.bitwise_and(edges, edges, mask=mask)


    d_target,score_bg = ball_score(edges,test,ground_truth,tracking_object)                  #获得得分

    if score_bg.size != 0 :
        if np.max(score_bg)>2:
            x, y, w, h = cv.boundingRect(d_target[list(score_bg).index(np.max(score_bg))])
            return [x, y, w, h]
    else:
        return None
def drawbox(test,tracking_object):
    if tracking_object==None:
        return 
    for key,value in tracking_object.items():
        bbox=value[0]
        cv.rectangle(test, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 0, 255), 2)


if __name__ == '__main__':
    # tracking_object={
    #     '1':[[230,145,150,378],0],
    #     '2':[[300,300,500,500],2]
    # }
    tracking_object=None
    cap = cv.VideoCapture('./video/offside1.mp4')
    ground_truth = cv.imread('./ball/ball.jpg')
    adaptive = 0
    while (cap.isOpened()):
        print('-----frame#-----')
        ret, test = cap.read()
        shrink = 2
        test = cv.resize(test, (test.shape[1]//shrink, test.shape[0]//shrink))
        st = time.time()
        # detect the ball ,output is [x, y, w, h]
        bbox = dective_by_background(test, ground_truth,tracking_object)
        if bbox is not None:
            print('has ball')
            if adaptive>=0:
                adaptive = adaptive - 0.5
            cv.rectangle(test, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
        else:
            print('no ball')
            adaptive = adaptive+1
        print('ada',adaptive)

        print('ft',time.time()-st)

        drawbox(test,tracking_object)
        test=cv.resize(test,(test.shape[1]//shrink, test.shape[0]//shrink))
        
        cv.imshow('test',test)
        cv.waitKey(1)
    cv.destroyAllWindows()






