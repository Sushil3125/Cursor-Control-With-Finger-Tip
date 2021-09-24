import cv2 as cv
import handTrackingModule as htm
import numpy as np
import time
import pyautogui

capture= cv.VideoCapture(0)
new_frame=1
prev_frame=0
detector=htm.HandDetector()

while True:
    success,frame=capture.read()
    frame=detector.findHand(frame, draw=True)
    lmlist=detector.findPosition(frame, draw=False)

    if len(lmlist)!=0:
        cv.circle(frame,(lmlist[8][1],lmlist[8][2]),15,(99,255,25),thickness=-1)
        x2,y2=lmlist[8][1],lmlist[8][2]
        # pyautogui.moveTo(int(x2*3),int(y2*2.25))

    new_frame=time.time()
    fps=1/(new_frame-prev_frame)
    prev_frame=new_frame

    cv.putText(frame,f'FPS: 30', (30,30), cv.FONT_HERSHEY_TRIPLEX, 1, (100,255,250), thickness=1)
    cv.imshow('Frame', frame)

    if cv.waitKey(20) & 0xFF==ord('d'):
        break
