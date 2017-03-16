import cv2
import cv
import os
import Image
import numpy




img = cv2.imread(os.path.dirname(os.path.abspath(__file__))+'/Radiographs/01.tif')
r = 1200.0 / img.shape[1]
dim = (1200, int(img.shape[0] * r))
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
cv2.imshow('img',resized)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

cv2.waitKey(0)

print 'volgende afbeelding'
temp = cv2.GaussianBlur(img,(5,5),5)
#temp = cv2.medianBlur(img,7)
img = cv2.addWeighted(img, 1.5, temp, -0.5, 0, img)
img = clahe.apply(img)
r = 1200.0 / img.shape[1]
dim = (1200, int(img.shape[0] * r))
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
cv2.imshow('img',resized)
cv2.waitKey(0)

img = cv2.medianBlur(img,11)



r = 1200.0 / img.shape[1]
dim = (1200, int(img.shape[0] * r))
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
cv2.imshow('img',resized)
