import cv2
from matplotlib.pyplot import gray
import numpy as np
from torch import scalar_tensor

image = np.uint8(cv2.imread('people1.jpg'))
print(image.shape)
#image = cv2.resize(image,(800,600))
gray =cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#cv2.imshow('temp',np.uint8(gray))
#cv2.waitKey(0)
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
detection = face_detector.detectMultiScale(gray,scaleFactor = 1.1, minSize = (60,60) )
eye_detector = cv2.CascadeClassifier('haarcascade_eye.xml')
detection_eye = eye_detector.detectMultiScale(gray,scaleFactor = 1.1, minNeighbors= 15, minSize=(10,10), maxSize = (60,60)) 
print(detection)
for (x,y,w,h) in detection:
    #print(x,y,h,w)
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
for (x,y,w,h) in detection_eye:
    #print(x,y,h,w)
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
cv2.imshow('temp',np.uint8(image))
cv2.waitKey(0)
'''
image2 = np.uint8(cv2.imread('people2.jpg'))
print(image2.shape)
gray2 =cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
face_detector2 = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
detection2 = face_detector.detectMultiScale(gray2,scaleFactor = 1.2, minNeighbors= 7, minSize=(10,10))#,
                                            #maxSize = (200,250) )
print(detection2)
for (x,y,w,h) in detection2:
    #print(x,y,h,w)
    cv2.rectangle(image2,(x,y),(x+w,y+h),(0,255,0),2)

cv2.imshow('temp',np.uint8(image2))
cv2.waitKey(0)
'''