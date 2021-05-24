
#-----------------------
#    siamFC++、识别、kalman滤波结合识别球
#    2020.11.24
#-----------------------
import argparse
import math
import sys
import time

import cv2 as cv
import imutils
import numpy as np
from imutils import paths
from matplotlib import pyplot as plt
from skimage import io
from skimage.metrics import structural_similarity

from ball.detect_by_edge import ball_score, dective_by_background
from ball.kalmanfilter import KalmanBoxTracker
from ball.Localdetectball import Localdective_by_background,RegionDetectBall

from siamfcpp.Tracker import SiamFCppTracker
from siamfcpp.model_build import build_model
from edgeline import offside_dectet 

#from tracking_camera import init_siam,initBBinit,siam_follow

import torch
import os
from math import sqrt,cos
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def chack_wh(pred):
    w=pred[2]
    h=pred[3]
    if w*h>150 and (w/h>1.3 or h/w>1.3):
        return False
    else:
        return True

def check_it_by_kalman(pred_by_Kalman,pred_by_background):
    dex=abs(pred_by_Kalman[0]-pred_by_background[0])
    dey=abs(pred_by_Kalman[1]-pred_by_background[1])
    # 检测面积和长宽比
    if chack_wh(pred_by_background):
        if dex<30 and dey<30:
            #检查kalman预测和background之间差距
            return True
        else:
            return True#False
    else:
        return False

def check_it_by_track(pred,track_it,count,th=35):
    if len(track_it)<=2:
        return True
    else:
        dx=abs(track_it[-1][0]-pred[0])
        dy=abs(track_it[-1][1]-pred[1])
        if dx>th*count or dy>th*count:
            print(str(pred)+'   shanchul**************')
            return False
        else:
            return True


def sim_it(ground_truth,temp,test):
    #比较相似度
    for i in range(4):
        temp[i]=max(0,temp[i])
    temp = cv.resize(test[temp[1]:temp[1]+temp[3],temp[0]:temp[0]+temp[2]], (ground_truth.shape[1], ground_truth.shape[0]))
    sim = structural_similarity(ground_truth, temp, multichannel=True)
    print(sim)
    if sim>0.5:
        return True
    else:
        return False

def ball_touch(track_it,track_object):
    meg=5
    if len(track_it)<=4:
        return False
    else:
        v_now=np.array([track_it[-1][0]-track_it[-2][0],track_it[-1][1]-track_it[-2][1]])
        v_before=np.array([track_it[-2][0]-track_it[-3][0],track_it[-2][1]-track_it[-3][1]])
        cos=(v_now[0]*v_before[0]+v_now[1]*v_before[1])/(np.linalg.norm(v_now)*np.linalg.norm(v_before)+0.001)

        if cos<0.9:# 有可能发生触球
            if track_object is not None:# 有人交互，进一步判断
                for i in track_object.keys():
                    if track_object[i][1]!=1:# 分类要是人
                        (x,y,w,h)=[int(i) for i in track_object[i][0]]
                        cx,cy=track_it[-1][0]+track_it[-1][2]//2,track_it[-1][1]+track_it[-1][3]//2
                        if max(0,(x-meg))<=cx<=min(1920,(x+w+meg)) and max(0,(y-meg))<cy<min(1080,(y+h+meg)):# 中心在人框某个范围内
                            return True  
            else:# 否则小于0.9就认为是触球了
                print('tracking_object is None')
                return True
        return False

def ball_track(balldataqueue,ballresultqueue,cap=None):
    # init
    pred=None
    pred_by_siam=None 
    pred_by_Kalman=None 
    pred_by_background=None
    tracking_object=None
    track_it=[]         #取信的路径
    ground_truth = cv.imread('./ball/ball2.jpg')
    state=None
    shrink = 2  
    num_of_break_frame=50 
    test=None
    #break_down=False
    count_lost_frame=1
    debug=True
    touch=False

    Model = build_model("siamfcpp/models/siamfcpp-tinyconv-vot.pkl")
    SiamTracker=SiamFCppTracker()
    SiamTracker.set_model(Model)

    # 初始化循环
    while True:
        # 读取
        if cap==None:
            print("get ball")
            try:
                test,tracking_object=balldataqueue.get()
            except Exception as Err:
                tracking_object=None
                print(Err)
                continue
        else:
            r,test=cap.read()
            test = cv.resize(test, (1920,1080))
        pred=dective_by_background(test,ground_truth,tracking_object=tracking_object)   #背景检测法
        if pred is None:
            ballresultqueue.put([pred,touch])
            continue
        SiamTracker.init(test,pred)
        Kalman=KalmanBoxTracker(np.array(pred))
        track_it.append(pred)
        if debug:
            print("background 1",pred)
        if pred!=None and cap==None:
            ballresultqueue.put([pred,touch])
        if test is not None and cap!=None:
            try:
                x,y,w,h=pred
                cv.rectangle(test,(x,y),(x+w,y+h),[0,0,255],3)
            except Exception as Err:
                print(Err)
            cv.imshow('t',test)
            cv.waitKey(-1)

        # 进入正式循环
        count=1
        while True:
            k=None
            goon=True
            # 读取
            if cap==None:
                try:
                    test,tracking_object=balldataqueue.get()
                    if test.shape[1]!=1080:
                        test = cv.resize(test, (1920,1080))
                except Exception as Err:
                    tracking_object=None
                    print(Err)
                    continue
            else:
                r,test=cap.read()
                test = cv.resize(test, (1920,1080))

            count+=1
            # 打断机制
            if count>=num_of_break_frame:
                pred_by_background=dective_by_background(test,ground_truth,tracking_object=tracking_object)
                if pred_by_background is None:
                    goon=True
                else:
                    print("count_lost_frame:",count_lost_frame)
                    if check_it_by_track(pred_by_background,track_it,count_lost_frame):
                        count=1
                        pred=pred_by_background
                        SiamTracker.init(test,pred)
                        Kalman=KalmanBoxTracker(np.array(pred))
                        goon=False
                        if debug:
                            print('break and init siam/kalman')
                    else: 
                        goon=True
                        count=count-10   

            # 分支路径
            if goon:
                pred_by_siam,lost,pos=SiamTracker.update(test)
                pred_by_siam=[int(i) for i in pred_by_siam]
                
                Kalman.predict()
                pred_by_Kalman=[int(i) for i in Kalman.get_state()[:4].reshape(1,4)[0].tolist()]

                if lost:
                    # siamFC预测有问题时处理 siam丢失
                    pred_by_background=RegionDetectBall(test,ground_truth,target_pos=pos) # 检测区域背景识别
                    if pred_by_background is None:
                        count_lost_frame+=1
                        pred_by_background=dective_by_background(test,ground_truth, tracking_object=tracking_object)# 区域背景检测失败，调用全局背景检测
                        if pred_by_background is None:# 全局背景也检测不到
                            pred = pred_by_Kalman
                            track_it.append(pred)
                            if debug:
                                print("kalman 2",pred,"siam:None RD:None GD:None->Kalman")
                        else:
                            # 全局背景检测到了，要用kalman预测结果检查
                            print("count_lost_frame:",count_lost_frame)
                            if check_it_by_track(pred_by_background,track_it,count_lost_frame):
                                pred=pred_by_background
                                SiamTracker.init(test,pred)
                                Kalman.update(np.array(pred))
                                track_it.append(pred)
                                if debug:
                                    print("background",pred,"siam:None RD:None GD:Yes->GD")
                            else:
                                pred=pred_by_Kalman
                                track_it.append(pred)
                                if debug:
                                    print("kalman",pred,"siam:None RD:None GD:Yes but check wrong->Kalman")
                    else:
                        # 区域检测跟踪到了，需要检查是不是噪点
                        print("count_lost_frame:",count_lost_frame)
                        if check_it_by_track(pred_by_background,track_it,count_lost_frame):
                            pred=pred_by_background
                            SiamTracker.init(test,pred)
                            Kalman.update(np.array(pred))
                            track_it.append(pred)
                            if debug:
                                print("RDbackground 3",pred,"siam:None RD:Yes->RD")
                        else:
                            count_lost_frame+=1
                            pred_by_background=dective_by_background(test,ground_truth, tracking_object=tracking_object)# 觉得区域检测有问题，用全局检测
                            if pred_by_background is None:
                                pred = pred_by_Kalman
                                track_it.append(pred)
                                if debug:
                                    print("kalman 2-2",pred,"siam:None RD:Yes but check wrong GD:None->kalman")
                            else:
                                print("count_lost_frame:",count_lost_frame)
                                if check_it_by_track(pred_by_background,track_it,count_lost_frame):
                                    pred=pred_by_background
                                    SiamTracker.init(test,pred)
                                    Kalman.update(np.array(pred))
                                    track_it.append(pred)
                                    if debug:
                                        print("background",pred,"siam:None RD:Yes but check wrong GD:Yes->GD ")
                                else:
                                    pred=pred_by_Kalman
                                    track_it.append(pred)
                                    if debug:
                                        print("kalman",pred,"siam:None RD:Yes but check wrong GD:Yes but check wrong->Kalman")
                else:
                    count_lost_frame=max(1,count_lost_frame-1)
                    print("count_lost_frame:",count_lost_frame)
                    if sim_it(ground_truth,pred_by_siam,test):# siam预测无问题，但是有可能出错时 检测相似度
                        # 检测正确时 用siam的值作为真 更新kalman
                        Kalman.update(np.array(pred_by_siam))
                        pred=pred_by_siam
                        track_it.append(pred)
                        if debug:
                            print("siam 4",pred,"siam:Yes ->siam")
                    else:
                        # 调用区域检测背景检测
                        pred_by_background=RegionDetectBall(test,ground_truth,target_pos=pos)
                        # 这个时候区域检测可能检测不到
                        if pred_by_background is None :
                            # 全局检测
                            pred_by_background=dective_by_background(test,ground_truth,tracking_object=tracking_object) 
                            if pred_by_background is None:# 全局还是检测不到
                                pred=pred_by_siam
                                track_it.append(pred)
                                if debug:
                                   print("siam 5",pred,"siam:yes but sim wrong RD:None GD:None->siam")
                            else:
                                print("count_lost_frame:",count_lost_frame)
                                if check_it_by_track(pred_by_background,track_it,count_lost_frame):
                                    pred=pred_by_background
                                    SiamTracker.init(test,pred)
                                    Kalman.update(np.array(pred))
                                    track_it.append(pred)
                                    if debug:
                                        print("background",pred,"siam:yes but sim wrong RD:None GD:Yes->GD")
                                else:
                                    pred=pred_by_Kalman
                                    track_it.append(pred)
                                    if debug:
                                        print("kalman 7",pred,"siam:yes but sim wrong RD:None GD:Yes but check wrong->Kalman")#检查全局
                        else:
                            print("count_lost_frame:",count_lost_frame)
                            if check_it_by_track(pred_by_background,track_it,count_lost_frame):
                                pred=pred_by_background
                                SiamTracker.init(test,pred)
                                Kalman.update(np.array(pred))
                                track_it.append(pred)
                                if debug:
                                    print("RDbackground",pred,"siam:yes but sim wrong RD:Yes->RD")
                            else:
                                pred=pred_by_Kalman
                                track_it.append(pred)
                                if debug:
                                    print("kalman",pred,"siam:yes but sim wrong RD:Yes but check wrong->Kalman")#检测区域检测

            touch=ball_touch(track_it,track_object=None)
            #print('ball_cos:',touch,"#"*30)

            if cap==None:
                ballresultqueue.put([pred,touch])
            else:
                if test is not None:
                    try:
                        x,y,w,h=pred
                        cv.rectangle(test,(x,y),(x+w,y+h),[0,0,255],3)
                    except Exception as Err:
                        print(Err)
                    cv.imshow('t',test)
                    cv.waitKey(-1)


if __name__=='__main__':
    import torch
    torch.multiprocessing.set_start_method(method='spawn')
    cap = cv.VideoCapture('./video/offside1.mp4')
    box={
        #'1':[[230,145,150,378],0],
        #'2':[[300,300,500,500],2]
    }
    balldatequeue=torch.multiprocessing.Queue()
    ballresultqueue=torch.multiprocessing.Queue()
    ballTrack=torch.multiprocessing.Process(target=ball_track,args=(balldatequeue,ballresultqueue))
    ballTrack.start()
    # ball_track(None,None,cap)
    
    while cap:
        print('biglooping')
        r,f=cap.read()
        f = cv.resize(f, (1920,1080))
        balldatequeue.put([f,box])
        
        try:
            yyy=ballresultqueue.get()
        except RuntimeError:
            print("lost")
        except Exception as Err:
            print('111')
        else:
            if yyy is not None:
                x,y,w,h=yyy[0],yyy[1],yyy[2],yyy[3]
        try:
            cv.rectangle(f,(x,y),(x+w,y+h),[0,0,255],3)
            print((x,y,w,h))
        except Exception as E:
            print(E)
            
        cv.imshow('ooo',f)
        cv.waitKey(1)