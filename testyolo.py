import torch
import argparse
from PIL import Image
import cv2
from yolo.yolo import YOLO
import numpy as np
import imutils

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", type=str, default="../SRTP/video/DJI_0266.MP4",
                    help="path to input video file")
    args = vars(ap.parse_args())
    vs = cv2.VideoCapture(args["video"])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLO()

    while True:
        frame = vs.read()
        frame = frame[1] if args.get("video", False) else frame
        # print(frame.shape)
        # frame = cv2.resize(frame, (frame.shape[1]//2,frame.shape[0]//2))
        frame = imutils.resize(frame, height=frame.shape[0]//2, width=frame.shape[1]//2)
        frame = Image.fromarray(np.uint8(frame))
        frame = model.detect_image(frame)
        frame = cv2.cvtColor(np.asarray(frame),cv2.COLOR_RGB2BGR)
        cv2.imshow("result",frame)
        cv2.waitKey(1)
    
    