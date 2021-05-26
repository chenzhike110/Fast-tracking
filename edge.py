import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import time
# from skimage.morphology import skeletonize

# 使用霍夫直线变换做直线检测，前提条件：边缘检测已经完成

# 统计概率霍夫线变换
def offside_dectet(image, direction, ofplayers, dfplayer):

    debug = 0
    img_origin = image.copy()
    shrink1 = 2
    shrink2 = 4
    if direction in ['left','up']:
        dfplayer_x = dfplayer[0]
        dfplayer_y = dfplayer[1]
    else:
        dfplayer_x = dfplayer[0]+dfplayer[2]
        dfplayer_y = dfplayer[1]+dfplayer[3]

    image = cv2.resize(image,(math.ceil(image.shape[1]/shrink1),math.ceil(image.shape[0]/shrink1)))
    has_offside = []
    th = 30  # 边缘检测后大于th的才算边界

    gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    gray_origin = cv2.cvtColor(img_origin, cv2.COLOR_BGRA2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    x = cv2.Sobel(gray, cv2.CV_16S, 1, 0)  # x方向梯度
    y = cv2.Sobel(gray, cv2.CV_16S, 0, 1)  # y方向梯度
    absX = cv2.convertScaleAbs(x)  # 转回uint8
    absY = cv2.convertScaleAbs(y)
    edges = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)  # 各0.5的权重将两个梯度叠加
    dst, edges = cv2.threshold(edges, th, 255, cv2.THRESH_BINARY)  # 大于th的赋值255（白色）

    edges = cv2.resize(edges, (math.ceil(image.shape[1] / shrink2), math.ceil(image.shape[0] / shrink2)))


    if debug == 1:
        cv2.imshow('edge', edges)

    # 函数将通过步长为1的半径和步长为π/180的角来搜索所有可能的直线
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=min(edges.shape[0], edges.shape[1])/2,
                            maxLineGap=math.ceil(40/shrink2))
    # print(lines)
    angle = []  # 备选线的角度
    b = []  # 备选线 y=kx+b的b

    if lines is None:
        has_line = 0
    else:
        has_line = 1
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle_per = math.atan((y2 - y1) / (x2 - x1))  # 角度
            if angle_per < -np.pi / 4:  # 将角度换到-pi/4 ~ 3pi/4
                angle_per = angle_per + np.pi
            angle.append(angle_per)
            # b.append(x1 * (y2 - y1) / (x2 - x1) - y1)
            b.append(shrink1*shrink2*(x2*y1-x1*y2) / (x2 - x1) )
            if debug == 1:
                cv2.line(img_origin, (x1*shrink1*shrink2, y1*shrink1*shrink2), (x2*shrink1*shrink2, y2*shrink1*shrink2), (0, 0, 255), 1)  # 画线

        angle = np.array(angle)
        b = np.array(b)
        threshold = 0.3
        if direction=='up' or direction=='down':
            angle_delete_vertical = angle[(angle<threshold) & (angle>-threshold)]
            b_delete_vertical = b[(angle<threshold) & (angle>-threshold)]
        elif direction == 'left' or direction=='right':
            angle_delete_vertical = angle[(angle < np.pi/2 + threshold) & (angle > np.pi/2 - threshold)]
            b_delete_vertical = b[(angle < np.pi/2 + threshold) & (angle > np.pi/2 - threshold)]


        # print(angle_delete_vertical)
        # angle_ave = np.median(angle_delete_vertical)  # 角度平均值
        angle_ave = np.median(angle_delete_vertical)  # 角度中位数
        angle_diff = angle_delete_vertical - angle_ave # 与平均值的差
        b = b_delete_vertical[abs(angle_diff) < 0.08]  # 去除离群点
        angle = angle_delete_vertical[abs(angle_diff) < 0.08]  # 去除离群点
        # print(angle)

        if len(angle) == 0:
            has_line = 0
        else:
            k_unsort = np.tan(angle)  # 角度对应的k
            b = np.array(b)
            dis = abs(k_unsort * dfplayer_x - dfplayer_y + b) / np.sqrt(1 + k_unsort * k_unsort)  # 防守球员到线的距离
            dis = list(dis)
            angle_final = angle[dis.index(min(dis))]  # 选择离防守球员最近的线
            if abs(angle_final) < 0.001:  # 处理奇异情况
                if angle_final < 0:
                    angle_final = -0.001
                else:
                    angle_final = 0.001
            elif abs(angle_final) > 1.56 and abs(angle_final) < 1.58:
                angle_final = 1.56 * angle_final / abs(angle_final)
            k = np.tan(angle_final)  # 最终的k

            # k = 0.10422

            for ofplayer in ofplayers:
                if direction in ['left', 'up']:
                    ofplayer_x = ofplayer[0]
                    ofplayer_y = ofplayer[1]
                else:
                    ofplayer_x = ofplayer[0] + ofplayer[2]
                    ofplayer_y = ofplayer[1] + ofplayer[3]
                # # 画出越位线
                # y1_draw = int(dfplayer_y - k * dfplayer_x)
                # y2_draw = int(k * gray_origin.shape[1] - k * dfplayer_x + dfplayer_y)
                # if debug==1:
                #     cv2.line(img_origin, (0, y1_draw), (gray_origin.shape[1], y2_draw), (0, 255, 0), 1)
                #     # 画出防守球员和进攻球员
                #     cv2.circle(img_origin, (dfplayer_x, dfplayer_y), 5, (255, 0, 0))
                #     cv2.circle(img_origin, (ofplayer_x, ofplayer_y), 5, (255, 0, 0))

                # 越位判罚
                line_x = ofplayer_y - (dfplayer_y - k * dfplayer_x) / k
                line_y = k * ofplayer_x - k * ofplayer_x + ofplayer_y
                if direction == 'left':
                    if line_x > ofplayer_x:
                        has_offside.append(1)
                    else:
                        has_offside.append(0)
                elif direction == 'right':
                    if line_x < ofplayer_x:
                        has_offside.append(1)
                    else:
                        has_offside.append(0)
                elif direction == 'up':
                    if line_y < ofplayer_y:
                        has_offside.append(1)
                    else:
                        has_offside.append(0)
                elif direction == 'down':
                    if line_y > ofplayer_y:
                        has_offside.append(1)
                    else:
                        has_offside.append(0)
        if debug == 1:
            cv2.imshow("line_detect_possible_demo", img_origin)
    return k,has_line, has_offside

def draw_offside_line(img_origin,direction,dfplayer,k):
    debug = 1
    if direction in ['left','up']:
        dfplayer_x = dfplayer[0]
        dfplayer_y = dfplayer[1]
    else:
        dfplayer_x = dfplayer[0]+dfplayer[2]
        dfplayer_y = dfplayer[1]+dfplayer[3]

    # 画出越位线
    y1_draw = int(dfplayer_y - k * dfplayer_x)
    y2_draw = int(k * img_origin.shape[1] - k * dfplayer_x + dfplayer_y)
    if debug == 1:
        cv2.line(img_origin, (0, y1_draw), (img_origin.shape[1], y2_draw), (0, 255, 0), 1)
        # 画出防守球员和进攻球员
        cv2.circle(img_origin, (dfplayer_x, dfplayer_y), 5, (255, 0, 0))
        # cv2.circle(img_origin, (ofplayer_x, ofplayer_y), 5, (255, 0, 0))
    return img_origin

if __name__ == "__main__":
    cap = cv2.VideoCapture('/home/jiangcx/桌面/足球视频/offside2.mp4')
    while (cap.isOpened()):
        print('-----frame#-----')
        ret, img = cap.read()
        # img = cv2.imread('edge.png')
        img = cv2.resize(img,(1920,1080))
        # cv2.imshow('original', img)
        # img = cv2.imread('edge16.png')

        # （图像，向哪个方向进攻(left & right)，进攻球员x，进攻球员y，防守球员x，防守球员y
        start_time = time.time()
        ofplayers = np.array([[10,20]])
        deplayer = np.array([100, 200])
        k,has_line, has_offsides = offside_dectet(img, 'up', ofplayers, deplayer)
        draw_offside_line(img, "up", deplayer, k)
        time1 = time.time()
        print('time1', time1 - start_time)
        # time2 = time.time()
        # print('time2', time2- time1)
        for has_offside in has_offsides:
            if has_line == 1:
                print('has_line')
                if has_offside == 1:
                    print('越位')
                else:
                    print('不越位')
            else:
                print("no_line")

        cv2.waitKey(-1)
    cv2.destroyAllWindows()
