import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import math
import time
import tensorflow as tf
from helper_functions import preprocess_filename

model = tf.keras.models.load_model("./Models/model3/model3.h5")
cap = cv2.VideoCapture(1)
detector = HandDetector(minTrackCon=0.5, maxHands=1)

imgSize = 300
offset = 20
counter = 0

classes = ['A', 'B', 'C', 'D', "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O"]
while True:
    success, img = cap.read()
    # img = cv2.flip(img, 1)
    hands, img = detector.findHands(img, flipType=True)
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
                cv2.imwrite("test.jpg", imgWhite)
                test_img = preprocess_filename("test.jpg")
                # print(test_img)
                preds = model.predict(tf.expand_dims(test_img, axis= 0))
                max_preds = preds[0].argmax()
                percentage = preds[0][max_preds] * 100
                label = classes[max_preds]
                print(percentage)
                cv2.putText(img, f'{percentage:.2f}',(x + 70, y - 20 ) , cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
                cv2.putText(img, f'{label}',(x + 70 + 30 , y - 50 ) , cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)

                print(label)


            else:
                k = imgSize/w
                hCal = math.ceil(k*h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize-hCal)/2)
                imgWhite[hGap:hCal + hGap,:] = imgResize
                cv2.imwrite("test.jpg", imgWhite)
                test_img = preprocess_filename("test.jpg")
                # print(test_img)
                preds = model.predict(tf.expand_dims(test_img, axis=0))
                max_preds = preds[0].argmax()
                percentage = preds[0][max_preds] * 100
                label = classes[max_preds]
                print(percentage)
                cv2.putText(img, f'{percentage:.2f}', (x + 70, y - 20), cv2.FONT_HERSHEY_PLAIN, 3,
                            (0, 0, 255), 3)
                cv2.putText(img, f'{label}', (x + 70 + 30 , y - 50 ), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)
                print(label)

            cv2.imshow('ImageCrop', imgCrop)
            cv2.imshow('ImageWhite', imgWhite)
        except Exception as e:
            print(e)
            pass
    cv2.imshow('Testing', img)
    key = cv2.waitKey(1)
    if key == ord('q') :
        break
cap.release()
cv2.destroyAllWindows()
