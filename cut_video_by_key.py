import cv2 



path='./noneoffside259.mp4'
cap=cv2.VideoCapture(path)
shrink=1


r,frame=cap.read()
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
#size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
#frame=frameflow.frame
size=frame.shape[:2]
out = cv2.VideoWriter('./nooffside_259.mp4', fourcc, fps, (size[1],size[0]))

while cap:
    r,test=cap.read()
    #f = cv2.resize(test, (test.shape[1]//shrink, test.shape[0]//shrink))
    #f=cv2.resize(f,width=1080)
    cv2.imshow('k', cv2.resize(test,(1920,1080))) #press k to save the frame
    cv2.waitKey(1)
    if cv2.waitKey()==ord('k'):
        out.write(test)
    if cv2.waitKey()==ord('q'):
        break	
cap.release()
out.close()
