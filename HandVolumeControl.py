import cv2
import mediapipe as mp
from pyautogui import press
from pyautogui import hotkey
import time
# import numpy as np
from math import hypot as hypo
from ctypes import cast,POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

#------------------------#
wCam,hCam = 800,480
#------------------------#

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("Aitrainer.mp4")
cap.set(3,wCam)
cap.set(4,hCam)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface,POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volRange = volume.GetVolumeRange()
# minVol = volRange[0]
# maxVol = volRange[1]

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1,min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# pTime = 0
# cTime = 0

count=0

while True:
    success, img = cap.read()
    img = cv2.flip(img,1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    lmList= []
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h,w,c = img.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                # print(id,cx,cy)
                lmList.append([id,cx,cy])
                
            if len(lmList)!=0:
                x0,y0 = lmList[0][1], lmList[0][2]
                x4,y4 = lmList[4][1], lmList[4][2]

                x8,y8 = lmList[8][1], lmList[8][2]
                x5,y5 = lmList[5][1], lmList[5][2]
                length85 = hypo(x8-x5,y8-y5)

                x12,y12 = lmList[12][1], lmList[12][2]
                x9,y9 = lmList[9][1], lmList[9][2]
                length129 = hypo(x12-x9,y12-y9)

                x16,y16 = lmList[16][1], lmList[16][2]
                x13,y13 = lmList[13][1], lmList[13][2]
                length1613 = hypo(x16-x13,y16-y13)

                x20,y20 = lmList[20][1], lmList[20][2]
                x17,y17 = lmList[17][1], lmList[17][2]
                length2017 = hypo(x20-x17,y20-y17)


                length48 = hypo(x4-x8,y4-y8)
                length412 = hypo(x4-x12,y4-y12)              

                palmlength= (hypo(x9-x0,y9-y0))/2
                # pyautogui.hotkey('alt', 'tab')


                if (count<1):
                    if ((length85>palmlength)and(length2017>palmlength)and(length129<palmlength)and(length1613<palmlength)):
                        press("space")
                        count=count+1                        
                    elif ((length85>palmlength)and(length2017<palmlength)and(length129<palmlength)and(length1613<palmlength)):
                        press("left")
                        count=count+1
                    elif ((length85<palmlength)and(length2017>palmlength)and(length129<palmlength)and(length1613<palmlength)):
                        press("right")
                        count=count+1
                    elif ((length85<palmlength)and(length2017<palmlength)and(length129>palmlength)and(length1613>palmlength)):
                        hotkey('alt','f4')
                        count=count+1
                if ((length85<palmlength)and(length2017<palmlength)and(length129<palmlength)and(length1613<palmlength)):
                        count=0
                
                # current = volume.GetMasterVolumeLevel()
                # if length48<25:
                #     hotkey('win','x')
                #     press("u")
                #     press("s")
                if length412<25:
                    volume.SetMasterVolumeLevel(-96.0 ,None)
                if length48<25:
                    volume.SetMasterVolumeLevel(0.0 ,None)
                
                                     
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # cTime = time.time()
    # fps = 1/(cTime-pTime)                FPS
    # pTime = cTime
    # cv2.putText(img, str(int(fps)), (10,40), cv2.FONT_HERSHEY_COMPLEX,1
    #             ,(255,0,255),2)
            
    cv2.imshow("Image", img)
    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        cap.release()
        break
