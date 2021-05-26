import math

import cv2
import cv2 as cv
import numpy as np
import time
from tqdm import tqdm
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity

# 统计概率霍夫线变换
def circle_dectet(image):
    penalty = []

    shrink = 2
    image = cv2.resize(image,(math.ceil(image.shape[1]/shrink),math.ceil(image.shape[0]/shrink)))
    th = 30  # 边缘检测后大于th的才算边界

    gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    x = cv2.Sobel(gray, cv2.CV_16S, 1, 0)  # x方向梯度
    y = cv2.Sobel(gray, cv2.CV_16S, 0, 1)  # y方向梯度
    absX = cv2.convertScaleAbs(x)  # 转回uint8
    absY = cv2.convertScaleAbs(y)
    edges = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)  # 各0.5的权重将两个梯度叠加

    shrink2 = 2
    edges = cv2.resize(edges, (math.ceil(edges.shape[1] / shrink2), math.ceil(edges.shape[0] / shrink2)))
    dst, edges = cv2.threshold(edges, th, 255, cv2.THRESH_BINARY)  # 大于th的赋值255（白色）
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges,kernel,iterations = 1)
    # edges = cv2.erode(edges,kernel,iterations= 2 )

    contours_person, hier = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for d in contours_person:
        x, y, w, h = cv2.boundingRect(d)
        if d.shape[0]<math.ceil(100/shrink2):
            edges[y:y+h,x:x+w] = np.zeros((h,w))

    # 霍夫圆变换
    # dp累加器分辨率与图像分辨率的反比默认1.5，取值范围0.1-10,越小越准
    dp = 2
    # minDist检测到的圆心之间的最小距离。如果参数太小，则除了真实的圆圈之外，还可能会错误地检测到多个邻居圆圈。 如果太大，可能会错过一些圈子。取值范围10-500
    minDist = 100
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp, minDist, param1=45, param2=10,
                                   minRadius=math.ceil(60 / (shrink*shrink2)), maxRadius=math.ceil(300 / (shrink*shrink2)))

    if circles is not None:
        circles = np.uint16(np.around(circles))
    else:
        return 0,[[0,0,0,0]]



    # c = []
    # for circle in circles[0]:
    #     # circle = circles[0]
    #     cir = np.zeros((edges.shape[0],edges.shape[1]))
    #     # 绘制外圆
    #     cv2.circle(cir, (circle[0], circle[1]), circle[2], (255, 255, 255), 1)
    #     # 绘制圆心
    #     # cv2.circle(cir, (circle[0], circle[1]), 2, (255, 255, 255), 2)
    #     cir = cv2.bitwise_and(cir,cir,mask=edges)
    #     c.append(len(cir[cir == 255])/(2*math.pi*circle[2]))
    # index = c.index(max(c))

    index = 0
    #只取中间的圆
    for circle in circles[0]:
        if abs(circle[0]-edges.shape[1]//2)>20 :
            if index<circles.shape[1]-1:

                index = index + 1
            else:
                # index = 0
                return 0, [[0, 0, 0, 0]]

        else:
            penalty.append([circle[0] * shrink * shrink2, circle[1] * shrink * shrink2,
                            circle[2] * shrink * shrink2, shrink * shrink2])
    # penalty.append([circles[0][index][0]*shrink*shrink2,circles[0][index][1]*shrink*shrink2,circles[0][index][2]*shrink*shrink2,shrink*shrink2])


    # cv2.imshow('edge', edges)

    return 1,penalty


def ball_score(gray_test,test,ground_truth,im_x,pos,debug=False):
    global adaptive
    adaptive = 0
    contours_person, hier = cv.findContours(gray_test, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    w_h = [] #长宽比
    c_s = [] #周长面积比
    ssim = [] #相似度
    d_target = [] #目标区域
    score = [] #得分

    # detect the circle
    has_circle, circles = circle_dectet(im_x)
    c_bound = 30
    for circle in circles:
        cv.circle(im_x, (circle[0], circle[1]), c_bound, (255, 0, 0), 2)
    circles = np.array(circles)
    x_c = circles[:,0]
    y_c = circles[:,1]

    # xywh = list(map(cv.boundingRect,contours_person))
    # xywh = np.array(list(map(np.array,xywh)))
    # wh = np.min(temp[:,0]/temp[:,1],temp[:,1]/temp[:,0])
    # print(temp.shape)
    if debug==True:
        contours_person=tqdm(contours_person)
    for d in tqdm(contours_person):
        if d.shape[0]<10:
            continue
        x, y, w, h = cv.boundingRect(d)     # 获取矩形框边界坐标
        # cv.rectangle(test, (x, y), (x + w, y + h), (0, 255, 0), 2)
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
        #print(pos,[x,y],x_c, x-gray_test.shape[0]//2+pos[0], y_c, y-gray_test.shape[0]//2+pos[1], has_circle, (np.abs(x_c - x +gray_test.shape[0]//2 - pos[0]) > c_bound).all(),(np.abs(y_c - y + gray_test.shape[0]//2- pos[1]) > c_bound).all())
        # np.max([2, 50 - adaptive]) <= area_obj < np.min([800, 100 + adaptive]) and np.max([2, 50 - adaptive]) < w * h < (100 + adaptive)
        if np.max([5,10]) <= area_obj <np.min([800,100]) and np.max([5,10])<w*h<np.min([800,100]) and cs>0.5 and wh>0.5 and  x > merge and y > merge and abs(x - test.shape[1])>merge and abs(x - test.shape[1])>merge and abs(y - test.shape[0])>merge and (has_circle == 0 or ((np.abs(x_c - x +gray_test.shape[0]//2 - pos[0]) > c_bound).all()  or (np.abs(y_c - y + gray_test.shape[0]//2- pos[1]) > c_bound).all())):

            # print(pos,x_c,x,y_c,y,has_circle,(np.abs(x_c-x-pos[0])>c_bound).all(),(np.abs(y_c-y-pos[1])>c_bound).all())
            # if not (has_circle == 0 or (np.abs(x_c-x)>c_bound).all()  and (np.abs(y_c-y)>c_bound).all()):
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
    # print('wh',w_h,'cs',c_s,'s',ssim)
    score = np.array(w_h)+np.array(c_s)+np.array(ssim)
    return d_target , score

def Localdective_by_background(test,ground_truth,im_x,pos):

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

    # cv.imshow('edge',cv.resize(edges,(edges.shape[1]//2,edges.shape[0]//2)))
    print('局部检测')
    st = time.time()
    d_target , score_bg = ball_score(edges,test,ground_truth,im_x,pos)                  #获得得分

    if score_bg.size != 0 :
        if np.max(score_bg)>2:
            x, y, w, h = cv.boundingRect(d_target[list(score_bg).index(np.max(score_bg))])
            return [x, y, w, h]
    else:
        return None


def RegionDetectBall(im_x,groundtruth,target_pos,cut_scale=100):
        if (int)(target_pos[0] - cut_scale)<0:
            cut_scale = target_pos[0]
        if (int)(target_pos[0] + cut_scale)>im_x.shape[1]:
            cut_scale = im_x.shape[1] - target_pos[0]
        if (int)(target_pos[1] - cut_scale) < 0:
            cut_scale = target_pos[1]
        if (int)(target_pos[1] + cut_scale) > im_x.shape[0]:
            cut_scale = im_x.shape[0] - target_pos[1]
        im_x_crop_show = im_x[max(0,(int)(target_pos[1] - cut_scale)):min((int)(target_pos[1] + cut_scale),im_x.shape[0]),
                         max(0,(int)(target_pos[0] - cut_scale)):min((int)(target_pos[0] + cut_scale),im_x.shape[1])]
        bbox = Localdective_by_background(im_x_crop_show,groundtruth,im_x,target_pos)# xywh
        if bbox is not None:
            print('has ball')
            bbox[0] = bbox[0]+target_pos[0]-cut_scale
            bbox[1] = bbox[1]+target_pos[1]-cut_scale
            bbox=[int(i) for i in bbox]
            return bbox
        else:
            print('no ball')
            return None


if __name__ == '__main__':

    cap = cv.VideoCapture('/home/jiangcx/桌面/足球视频/offside1.mp4')
    ground_truth = cv.imread('ball2.jpg')
    adaptive = 0
    while (cap.isOpened()):
        print('-----frame#-----')
        ret, test = cap.read()
        shrink = 2
        test = cv.resize(test, (test.shape[1]//shrink, test.shape[0]//shrink))
        st = time.time()
        # detect the ball ,output is [x, y, w, h]
        bbox = Localdective_by_background(test, ground_truth)
        if bbox is not None:
            print('has ball')
            if adaptive>=0:
                adaptive = adaptive - 0.5
            # cv.rectangle(test, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 1)
        else:
            print('no ball')
            adaptive = adaptive+1
        # print('ada',adaptive)

        # print('ft',time.time()-st)

        cv.imshow('test',cv.resize(test,(test.shape[1]//1,test.shape[0]//1)))
        cv.waitKey(-1)
    cv.destroyAllWindows()












