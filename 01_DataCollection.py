import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import math
import time
import os
import string
from helper_functions import create_folder


cap = cv2.VideoCapture(1)
detector = HandDetector(minTrackCon=0.5, maxHands=1)

create_folder()
imgSize = 300
offset = 20
counter = 0
gesture = input("Enter the Gesture: ").upper()
folder = f"./Data/{gesture}"
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        imgCrop = img[y - offset : y + h + offset, x - offset : x + w + offset]

        imgCropShape = imgCrop.shape

        try:
            aspectRatio = h/w
            if aspectRatio > 1:
                k = imgSize/h
                wCal = math.ceil(k*w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize-wCal)/2)
                imgWhite[:, wGap:wCal + wGap] = imgResize

            else:
                k = imgSize/w
                hCal = math.ceil(k*h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize-hCal)/2)
                imgWhite[hGap:hCal + hGap,:] = imgResize
            cv2.imshow('ImageCrop', imgCrop)
            cv2.imshow('ImageWhite', imgWhite)
        except:
            pass
    cv2.imshow('Data Collection', img)
    key = cv2.waitKey(1)
    if counter == 300:
        break
    if key == ord('s'):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
    elif key == ord('q') :
        break
cap.release()
cv2.destroyAllWindows()
