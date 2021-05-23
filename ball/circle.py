import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from skimage.morphology import skeletonize
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny



# 使用霍夫直线变换做直线检测，前提条件：边缘检测已经完成



# 统计概率霍夫线变换
def circle_dectet(image):
    penalty = []
    #t1 = time.time()
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


    #t2 = time.time()
    #print('t1:',t2-t1)


    # edges = skeletonize(edges / 255)
    # print(type(edges))
    # edges = edges.astype('uint8')*255

    # st = time.time()
    # lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 255, minLineLength=min(gray.shape[0], gray.shape[1])/3,
    #                        maxLineGap=20)
    # x_all = []
    # y_all = []
    # if lines is not None:
    #     for line in lines:
    #         x1, y1, x2, y2 = line[0]
    #         cv2.line(edges, (x1, y1), (x2, y2), (0, 0, 0), 1)  # 画线
    # #         x_all.append(x1)
    # #         x_all.append(x2)
    # #         y_all.append(y1)
    # #         y_all.append(y2)
    # #         # plt.imshow(img_line)
    # #         # plt.show()

    # REMOVE THE NOISE

        # area_obj = cv2.contourArea(d)
        # if area_obj / (w*h) >0.2:
        #     edges[y:y+h,x:x+w] = np.zeros((h,w))

    # print('4', time.time() - st)


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
    t3 = time.time()
    #print('t2:',t3-t2)
    # 霍夫圆变换
    # dp累加器分辨率与图像分辨率的反比默认1.5，取值范围0.1-10,越小越准
    dp = 2
    # minDist检测到的圆心之间的最小距离。如果参数太小，则除了真实的圆圈之外，还可能会错误地检测到多个邻居圆圈。 如果太大，可能会错过一些圈子。取值范围10-500
    minDist = 100
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp, minDist, param1=45, param2=10,
                                   minRadius=math.ceil(60 / (shrink*shrink2)), maxRadius=math.ceil(300 / (shrink*shrink2)))
    circles = np.uint16(np.around(circles))
    if circles is not None:
        circles = np.uint16(np.around(circles))
    else:
        return 0,[0,0,0,0]

    t4 = time.time()
    #print('t3:',t4-t3)
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
        if abs(circle[0]-edges.shape[1]//2)>20:
            if index<circles.shape[1]-1:
                #print(circles.shape[1])
                index = index + 1
            else:
                index = 0
        else:
            penalty.append([circle[0] * shrink * shrink2, circle[1] * shrink * shrink2,
                            circle[2] * shrink * shrink2, shrink * shrink2])
    penalty.append([circles[0][index][0]*shrink*shrink2,circles[0][index][1]*shrink*shrink2,circles[0][index][2]*shrink*shrink2,shrink*shrink2])


    # cv2.imshow('edge', edges)

    return 1,penalty




if __name__ == "__main__":
    cap = cv2.VideoCapture('4k.mp4')
    while (cap.isOpened()):
        start_time = time.time()
        print('-----frame#-----')
        ret, img = cap.read()
        img = cv2.resize(img,(img.shape[1]//2,img.shape[0]//2))

        time1 = time.time()
        print('time1', time1 - start_time)
        #
        has_circle,circles = circle_dectet(img)
        a = np.array(circles)
        print(((a[:,0]-1000)>0).all())


        print('time2',time.time()-time1)
        if has_circle==1:
            print('has_circle')
            for circle in circles:
                # 绘制外圆
                cv2.circle(img, (circle[0], circle[1]), circle[2], (255, 0, 0), 1)
                # 绘制圆心
                cv2.circle(img, (circle[0], circle[1]), 2, (255, 0, 0), 2)
        else:
            print('no_circle')
        cv2.imshow('img',img)
        cv2.waitKey(-1)
    cv2.destroyAllWindows()
