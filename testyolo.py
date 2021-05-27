import torch
import argparse
from PIL import Image
import cv2
from yolo.yolo import YOLO
import numpy as np
import imutils
# from edge import get_new

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", type=str, default="video/offside2_2.mp4",
                    help="path to input video file")
    args = vars(ap.parse_args())
    vs = cv2.VideoCapture(args["video"])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLO()

    while True:
        frame = vs.read()
        frame = frame[1] if args.get("video", False) else frame
        frame = imutils.resize(frame, height=1080//2, width=1920//2)
        # frame = get_new(frame)
        # print(frame.shape)
        frame = cv2.resize(frame, (frame.shape[1]//2,frame.shape[0]//2))
        frame = Image.fromarray(np.uint8(frame))
        # print(np.array(frame).shape)
        frame = model.detect_image(frame)
        # frame = np.array(frame)[:,:,:-1]
        frame = cv2.cvtColor(np.asarray(frame),cv2.COLOR_RGB2BGR)
        cv2.imshow("result",frame)
        cv2.waitKey(1)
    
    