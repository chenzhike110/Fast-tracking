import numpy as np
first_ball = 0

def cal_iou(box_ball,box_player):
    area_a = (box_ball[:,2]- box_ball[:,0])* (box_ball[:,3]-box_ball[:,1])
    area_b = (box_player[:,2] -box_player[:,0]) * (box_player[:,3] - box_player[:,1])

    w = np.minimum(box_player[:,2] ,box_ball[:,2]) - np.maximum(box_ball[:,0],box_player[:,0])
    h = np.minimum(box_player[:,3] ,box_ball[:,3]) - np.maximum(box_ball[:,1],box_player[:,1])

    w = np.maximum(w,0)
    h = np.maximum(h,0)

    area_c = w * h

    return area_c / (area_a + area_b - area_c)

def ball_state(frame,boxes, top_label):
    global first_ball #是否第一次检测到球 0：从未检测到球；1：检测到过球
    global old_box_ball #上一次检测到的球box
    touch_person = 0 #0：没接触，array（1，4）：接触球的人的box
    ball_boxes = boxes[top_label == 1] #识别出是球的boxes
    player_boxes = boxes[top_label == 0] #识别出是人的boxes

    loss_ball_edge = 5 #超参数：识别到球距离视野边界多少为将离开视线的状态
    if (top_label == 1).any(): #检测到球
        if ball_boxes.shape[0] != 1:
            state = 0 #检测到大于一个球
            print('two ball')
        else:                        #检测到两个球
            if first_ball == 0:         #从未检测到球的话把第一次检测到的球坐标给他
                old_box_ball = ball_boxes
                first_ball = 1
            else:                       #检测到过球则更新
                old_box_ball = ball_boxes

            iou = cal_iou(ball_boxes, player_boxes) #计算球和每个人的iou
            if (iou > 0).any():
                state = 1 #触球
                print('touch')
                touch_person = player_boxes[np.argmax(iou), :] #触球的人的box
            else:
                state = 2 #有球但未触球
                print('pass')
    else:
        if first_ball == 1 and (
                old_box_ball[:, 0] < loss_ball_edge or old_box_ball[:, 2] > frame.shape[0] - loss_ball_edge \
                or old_box_ball[:, 1] < loss_ball_edge or old_box_ball[:, 3] > frame.shape[1] - loss_ball_edge):
            state = 3 #球出边界所以没识别到
            print('ball out edge')
        elif first_ball == 0:
            state = 4 #还未第一次检测到球
            print('not find the first ball')
        else:
            state = 5 #球在场内但没识别到
            print('ball in court but loss')
    return state,touch_person
