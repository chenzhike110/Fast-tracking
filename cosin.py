import os
import random
import shutil
import time
from collections import Counter
from math import sqrt

import numpy as np
from cv2 import cv2 as cv
from matplotlib import pyplot as plt
import imutils

import datetime

class KNNClassifier:
    def __init__(self,video_name,modelpath=None):
        self.maskradio=2
        edgeThreshold=2
        patchSize=2
        self.orb = cv.ORB_create(edgeThreshold = edgeThreshold,patchSize=patchSize)
        self.video_name=video_name
        self.KNNModel=self.getKNNmodel(modelpath)

    def getKNNmodel(self,modelpath):
        KNNModel=[]
        for i in os.listdir(os.path.join(modelpath,self.video_name)):
            if i!=".DS_Store":
                for j in os.listdir(os.path.join(modelpath,self.video_name,i)):
                    if j!=".DS_Store":
                        img=cv.imread(os.path.join(modelpath,self.video_name,i,j))
                        m,_,_=self.main_color_moment(img)
                        k=[m,str(i)]
                        KNNModel.append(k)
        return KNNModel

    def prediction(self,box,frame,video_name,classes_name,padding,save_img_recode=False,k=9):
        img=frame[max(int(box[1]-padding),0):min(int(box[1]+box[3]+padding),frame.shape[0]),max(int(box[0]-padding),0):min(int(box[0]+box[2]+padding),frame.shape[1])]

        m,mat,box=self.main_color_moment(img)
        distancelist=[]
        pset=set()
        for i in self.KNNModel:
            d=self.distance(m,i[0])
            label=i[-1]
            distancelist.append([d,label])
            pset.add(label)
        
        distancelist=sorted(distancelist)
        pdict={}
        for i in pset:
            pdict[i]=0
        for i in range(k):
            pdict[distancelist[i][1]]+=1
        
        if save_img_recode:
            path='./knn_classes'
            video_name=video_name+'_pred'
            try:
                os.mkdir(os.path.join(path,video_name))
            except :
                try:
                    os.mkdir(os.path.join(path,video_name,max(pdict, key=pdict.get)))
                except :
                    pass
            p=os.path.join(path,video_name,max(pdict, key=pdict.get),''.join([str(i) for i in str(datetime.datetime.now()) if i.isdigit()]) +".jpg")
            print(p)
            try:
                cv.imwrite(p,img)
            except Exception as Error:
                print(Error)

        return classes_name.index(max(pdict, key=pdict.get)),mat,box#classes_name.index(max(pdict, key=pdict.get))#max(pdict, key=pdict.get)#,mat#classes_name.index(max(pdict, key=pdict.get))
    
    def distance(self,p1,p2):
        try:
            op2=np.linalg.norm(p1-p2)
        except Exception as Error:
            print('[distance wrong]'+Error)
        return op2
    
    def mini_img(self,img):
        kp = self.orb.detect(img, None)
        kp, _= self.orb.compute(img, kp)
        pointlist=[]
        mat=img.copy()
        for i in range(len(kp)):
            pointlist.append(list(map(int,kp[i].pt)))
            mat = cv.circle(mat, tuple(list(map(int,kp[i].pt))), self.maskradio, (0, 0, 255),-1)
        _,_,R=cv.split(mat)

        pointlist=sorted(pointlist,key=lambda x:x[0])
        try:
            xmin,xmax=pointlist[0][0],pointlist[-1][0]
            pointlist=sorted(pointlist,key=lambda x:x[1])
            ymin,ymax=pointlist[0][1],pointlist[-1][1]
            if xmax-xmin<=5:
                xmin-=2
                xmax+=2
            if abs(ymax-ymin)<=5:
                ymin-=2
                ymax+=2
            R=R[ymin:ymax,xmin:xmax]
            R=cv.dilate(R,(13,13))
            R=cv.erode(R,(19,19))
        except Exception as Err:
            print(Err)
            # cv.imshow('wrong',img)
            # cv.waitKey(0)

            #cv.destroyAllWindows()
        return xmin,xmax,ymin,ymax,R

    def main_color_moment(self,img,see_make=False)->list:

        #xmin,xmax,ymin,ymax,mat=self.mini_img(img)
        #img=img[ymin:ymax,xmin:xmax]
        mask=get_mask(img)

        img1=img.copy()
        img1=cv.cvtColor(img,cv.COLOR_BGR2HSV)
        img1[mask!=1]=[0,0,0]
        # B,G,R=cv.split(img1)
        # #hhh=img.copy()
        # R[R!=0]=0
        # G[G!=0]=0
        # hhh=cv.merge([B,G,R])
        if see_make:
            cv.imshow('rrr',mask)
            cv.imshow('ooo',img)
            cv.imshow('ppp',img1)
            #cv.imshow('yyy',hhh)
            #cv.imshow('uuu',mmm)
            cv.waitKey(0)                         
            cv.destroyAllWindows()

        # N=np.sum(mask)
        # R,G,B=cv.split(img1)
        # R=np.sum(R)/N
        # G=np.sum(G)/N
        # B=np.sum(B)/N
        # L=np.linalg.norm(np.array([R,G,B]))
        # return np.array([R,G,B,L]),None,None
        hist1 = cv.calcHist([img1],[0], None, [15], [1.0,255.0])
        hist2 = cv.calcHist([img1],[1], None, [3], [1.0,255.0])
        #hist3 = cv.calcHist([img1],[2], None, [5], [1.0,255.0])
        #print(hist6)

        hist1=hist1/np.sum(hist1)
        hist2=hist2/np.sum(hist2)
        #hist3=hist3/np.sum(hist3)
        hist=np.concatenate((hist1,hist2),axis=0)
        print(hist.shape)
        return hist,None,None#,mat,(xmin,xmax,ymin,ymax)

def init_get_video(classname,video_name,num_of_photo,path,update_data=False):
    flag=0
    try:
        os.mkdir(os.path.join(path,video_name))
    except Exception as Error:
        print(Error)
        flag=1
        for i in classname:
            try:
                length=len(os.listdir(os.path.join(path,video_name,i)))
            except Exception as Error:
                flag=0
                break
            if length<num_of_photo:
                flag=0
                break
        if update_data==True and flag==1:
            flag=0
    if flag:
        print('you have the dataset')    
        return True
    else:
        for i in classname:
            try:
                p=os.path.join(path,video_name,i)
                shutil.rmtree(p)
            except Exception as Error:
                print(Error)
                continue
        for i in classname:
            try:
                os.mkdir(os.path.join(path,video_name,i))
            except Exception as err:
                print(err)
                continue
    try:
        p=os.path.join(path,video_name+'_pred')
        shutil.rmtree(p)
    except :
        pass


def get_data_from_video(frame,box,classname,padding,video_name,path,num_of_photo=25):
    """
        从视频中获取前20帧的图像素材
    """

    img=frame[int(box[1]-padding):int(box[1]+box[3]+padding),int(box[0]-padding):int(box[0]+box[2]+padding)]
    p=os.path.join(path,video_name,str(classname),''.join([str(i) for i in str(datetime.datetime.now()) if i.isdigit()])+".jpg")
    print(p)
    if len(os.listdir(os.path.join(path,video_name,classname)))<num_of_photo:
        try:
            cv.imwrite(p,img)
            return p
        except Exception as Error:
            print(Error,p)
    else:
        return None



def mini_img(frame,yolo):
    result=[]
    for i in yolo:#[[x,y,w,h,c]]
        x,y,w,h,c=i[0],i[1],i[2],i[3],i[4]
        x,y=x-w//2,y-h//2
        x,y,w,h=int(x),int(y),int(w),int(h)
        img=frame[y:y+h,x:x+w]
        mask=get_mask(img)
        contours_person, hier = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        S=0
        for d in contours_person:
            S_1=cv.contourArea(d)
            if S<S_1:
                x_c, y_c, w_c, h_c = cv.boundingRect(d)
                if w/h<2 and h/w<2:
                    S=S_1
        print(S)
        if S<90:
            x_c,y_c=x_c-5,y_c-5
            w_c,h_c=w_c+10,h_c+10
        box=[x+x_c,y+y_c,w_c,h_c,c]
        box=[box[0]+box[2]//2,box[1]+box[3]//2,box[2],box[3],box[4]]
        result.append(np.array(box))
        #x,y,w,h,c=result[-1][0],result[-1][1],result[-1][2],result[-1][3],result[-1][4]
    return result

# def mini_img(frame,yolo,orb):
#     result=[]
#     for i in yolo:#[[x,y,w,h,c]]
#         x,y,w,h,c=i[0],i[1],i[2],i[3],i[4]
#         img=frame[y-h//2:y+h//2,x-w//2:x+w//2]
#         kp = orb.detect(img, None)
#         kp, _= orb.compute(img, kp)
#         pointlist=[]
#         for i in range(len(kp)):
#             pointlist.append(list(map(int,kp[i].pt)))
#         pointlist=np.array(pointlist)
#         pointlist=np.transpose(pointlist,axes=[1,0])
#         xmin,xmax=np.min(pointlist[0]),np.max(pointlist[0])
#         ymin,ymax=np.min(pointlist[1]),np.max(pointlist[1])
#         if xmax-xmin<=5:
#             xmin-=2
#             xmax+=2
#         if abs(ymax-ymin)<=5:
#             ymin-=2
#             ymax+=2
#         result.append(np.array([int(x-w//2+(xmin+xmax)//2),int(y-h//2+(ymin+ymax)//2),int(xmax-xmin),int(ymax-ymin),int(c)]))
#         x,y,w,h,c=result[-1][0],result[-1][1],result[-1][2],result[-1][3],result[-1][4]
#     return result
    

def len_all(path,videoname,classes_name):
    classes_photo=[]
    for i in classes_name:
        list_p=os.listdir(os.path.join(path,videoname,i))
        classes_photo.append(len(list_p))
    return classes_photo

def cos_in(img,point):
    height,width=img.shape[0],img.shape[1]
    filter=np.array([1,1,1])
    dot_result=img.copy()
    norm=img

    norm=norm.astype(np.int64)
    dot_result=dot_result.reshape(-1,3)
    dot_result=np.dot(dot_result,point)
    dot_result=dot_result.reshape(height,width,1)

    norm=norm.reshape(-1,3)
    norm=np.square(norm)
    norm=np.dot(norm,filter)
    norm=norm.reshape(height,width,1)
    norm=np.sqrt(norm)
    z=0.5*(dot_result/norm)/np.linalg.norm(point)+0.5
    return z

def normlization(img):
    '''
    img:float
    '''
    minx=np.min(img)
    return np.uint8(((img-minx)/(np.max(img)-minx))*255)

def softmax(img,iter=1):
    for i in range(iter):
        img=img/255
        img=np.exp(img)/np.sum(np.exp(img))
        img=normlization(img)
    return img

def get_mask(img,softmax_time=1,point=[80,140,140],th=30):
    H_call=cos_in(img,np.array(point))
    H_call=normlization(H_call)
    H_call=softmax(H_call,softmax_time)
    H_call=255-H_call
    H_call[H_call<th]=0
    H_call[H_call!=0]=1
    H_call=cv.erode(H_call,(5,5))
    H_call=cv.dilate(H_call,(5,5))
    return H_call

if __name__ == '__main__':
    online_data_save_path="knn_classes"
    classes_name = ["player", "ball", "team1", "team2", "judger"]

    # orb
    edgeThreshold=2
    patchSize=2

    see_make=False
    save_img_recode=True

    modelpath="./knn_classes"
    video_name="DJI_0273.MP4"
    classes_name = ["judger","team1","team2"]
    see_wrong=False#True
    see_make=False#True#False#True
    see_right=False
    save_img_recode=False
    del_pred=False

    if del_pred:
        try:
            p=os.path.join(modelpath,video_name+'_pred')
            shutil.rmtree(p)
        except :
            pass

    def test(test_path,KNN):
        fps=[]
        acc={}
        for i in os.listdir(test_path):
            if i!=".DS_Store":
                count=0
                wrong=0
                #print(i)
                for j in os.listdir(os.path.join(test_path,i)):
                    if j!=".DS_Store":
                        img=cv.imread(os.path.join(test_path,i,j))
                        stime=time.time()
                        fact,mat,box=KNN.prediction(img,video_name)
                        #print(fact)
                        #print(box)
                        endtime=time.time()
                        fps.append(1/(endtime-stime))
                        fact=classes_name[fact]
                        img=img[box[2]:box[3],box[0]:box[1]]
                        if see_right:
                            cv.imshow(str('true='+i+':'+fact),img)
                            cv.imshow(str('truemat='+i+':'+fact),mat)
                            cv.waitKey(0)
                            cv.destroyAllWindows()
                        if fact!=i:
                            wrong+=1
                            print(fact)
                            if see_wrong:
                                cv.imshow(str('true='+i+':'+fact),img)
                                cv.imshow(str('truemat='+i+':'+fact),mat)
                                img[mat!=255]=[0,0,0]
                                cv.imshow(str('jus'+i+':'+fact),img)
                                cv.waitKey(0)
                                cv.destroyAllWindows()
                        count+=1
                acc[i]=(count-wrong)/count
        print("fps={0:4>.2f}".format(sum(fps)/len(fps)))
        return acc

    KKK=KNNClassifier(video_name,modelpath)
    test_path=os.path.join(modelpath,video_name+'_test')#"./knn_classes/train_it2/"
    acc=test(test_path,KKK)
    print(acc)
