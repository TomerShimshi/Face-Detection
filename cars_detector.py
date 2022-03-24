import cv2
from matplotlib.pyplot import gray
import numpy as np
from torch import scalar_tensor

image = np.uint8(cv2.imread('car.jpg'))
print(image.shape)
#image = cv2.resize(image,(800,600))
gray =cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

cars_detector = cv2.CascadeClassifier('cars.xml')
detection = cars_detector.detectMultiScale(gray,scaleFactor = 1.02, minNeighbors= 6, minSize = (10,10) )

for (x,y,w,h) in detection:
    #print(x,y,h,w)
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

cv2.imshow('temp',np.uint8(image))
cv2.waitKey(0)
