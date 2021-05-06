import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import time


# 使用霍夫直线变换做直线检测，前提条件：边缘检测已经完成

# 统计概率霍夫线变换
def offside_dectet(image, direction, ofplayer_x, ofplayer_y, dfplayer_x, dfplayer_y):
    img_origin = image.copy()
    image = cv2.resize(image,(math.ceil(image.shape[1]/2),math.ceil(image.shape[0]/2)))
    has_offside = 0
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
    cv2.imshow('edge', edges)
    time2 = time.time()
    print('time2', time2 - time1)
    # 函数将通过步长为1的半径和步长为π/180的角来搜索所有可能的直线
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 255, minLineLength=min(gray.shape[0], gray.shape[1])/3,
                            maxLineGap=20)
    # print(lines)
    angle = []  # 备选线的角度
    b = []  # 备选线 y=kx+b的b

    time3 = time.time()
    print('time3', time3 - time2)

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
            b.append(x1 * (y2 - y1) / (x2 - x1) - y1)
            # cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 画线

        time4 = time.time()
        print('time4', time4- time3)

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
        b = b_delete_vertical[abs(angle_diff) < 0.02]  # 去除离群点
        angle = angle_delete_vertical[abs(angle_diff) < 0.02]  # 去除离群点
        # print(angle)
        if len(angle) == 0:
            has_line = 0
        else:
            k_unsort = np.tan(angle)  # 角度对应的k
            b = np.array(b)
            dis = abs(-k_unsort * dfplayer_x + dfplayer_y + b) / np.sqrt(1 + k_unsort * k_unsort)  # 防守球员到线的距离
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

            time5 = time.time()
            print('time5', time5 - time4)

            # 画出越位线
            y1_draw = int(dfplayer_y - k * dfplayer_x)
            y2_draw = int(k * gray_origin.shape[1] - k * dfplayer_x + dfplayer_y)
            cv2.line(img_origin, (0, y1_draw), (gray_origin.shape[1], y2_draw), (0, 255, 0), 1)
            # 画出防守球员和进攻球员
            cv2.circle(img_origin, (dfplayer_x, dfplayer_y), 5, (255, 0, 0))
            cv2.circle(img_origin, (ofplayer_x, ofplayer_y), 5, (255, 0, 0))

            # 越位判罚
            line_x = ofplayer_y - (dfplayer_y - k * dfplayer_x) / k
            line_y = k * ofplayer_x - k * ofplayer_x + ofplayer_y
            if direction == 'left':
                if line_x > ofplayer_x:
                    has_offside = 1
            elif direction == 'right':
                if line_x < ofplayer_x:
                    has_offside = 1
            elif direction == 'up':
                if line_y < ofplayer_y:
                    has_offside = 1
            elif direction == 'down':
                if line_y > ofplayer_y:
                    has_offside = 1

            time6 = time.time()
            print('time6', time6 - time5)

        cv2.imshow("line_detect_possible_demo", img_origin)
    return has_line, has_offside


if __name__ == "__main__":
    cap = cv2.VideoCapture('/home/jiangcx/桌面/足球视频/video5.mp4')
    while (cap.isOpened()):
        start_time = time.time()
        print('-----frame#-----')
        ret, img = cap.read()
        img = cv2.resize(img,(1920,1080))
        # cv2.imshow('original', img)
        # img = cv2.imread('edge16.png')
        time1 = time.time()
        print('time1', time1 - start_time)
        # （图像，向哪个方向进攻(left & right)，进攻球员x，进攻球员y，防守球员x，防守球员y
        has_line, has_offside = offside_dectet(img, 'up', 10, 20, 100, 200)
        # time2 = time.time()
        # print('time2', time2- time1)
        if has_line == 1:
            print('has_line')
            if has_offside == 1:
                print('越位')
            else:
                print('不越位')
        else:
            print("no_line")

        cv2.waitKey(1)
    cv2.destroyAllWindows()
