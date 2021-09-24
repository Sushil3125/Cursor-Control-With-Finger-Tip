import cv2 as cv
import mediapipe as mp
import time
    

class HandDetector():
    def __init__(self, mode=False, maxHand=2, detectionCon=0.5, trackingCon=0.5):
        self.mode=mode
        self.maxHand=maxHand
        self.detectionCon=0.5
        self.trackingCon=0.5
        


        self.mpHands= mp.solutions.hands
        self.hands=self.mpHands.Hands(self.mode, self.maxHand, 
                                self.detectionCon, self.trackingCon)
        self.mpDraw=mp.solutions.drawing_utils


    def findHand(self, frame, draw=True):


        imgRGB= cv.cvtColor(frame, cv.COLOR_BGR2RGB)    
        self.results= self.hands.process(imgRGB) 

        x1,y1,x2,y2=0,0,0,0


        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw==True:
                    self.mpDraw.draw_landmarks(frame, handLms, self.mpHands.HAND_CONNECTIONS)
        return frame

    
    def findPosition(self, frame, handNo=0, draw=True):

        lmList=[]
        if self.results.multi_hand_landmarks:
            myHand=self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id,lm)  #prints the lamdmark of all the 21 positions
                
                h, w, c = frame.shape
                cx, cy  = int(lm.x*w), int(lm.y*h)
                # print(id, cx, cy)
                lmList.append([id, cx, cy])    
                if draw==True:
                    cv.circle(frame, (cx,cy), 15, (255,0,0), thickness=-1)        
        return lmList

def main():
    capture= cv.VideoCapture(0)
    new_frame=0
    prev_frame=0
    detector=HandDetector()
    while True:
        success, frame= capture.read()
        frame=detector.findHand(frame)
        lmList=detector.findPosition(frame, draw=False)
        if len(lmList)!=0:
            print(lmList[4])

        new_frame=time.time()
        fps=1/(new_frame-prev_frame)
        prev_frame=new_frame

        
        cv.putText(frame,f'FPS: {int(fps)}', (30,30), cv.FONT_HERSHEY_TRIPLEX, 1, (100,255,250), thickness=1)
        cv.imshow('Frame', frame)

        if cv.waitKey(20) & 0xFF==ord('d'):
            break
 

    
if __name__=="__main__":
    main()