# -*- coding: utf-8 -*-

import cv2

cv2.namedWindow('original')
cv2.moveWindow('original', 0,0)

cv2.namedWindow('processed')
cv2.moveWindow('processed', 900,0)



img = cv2.imread('rgb/cps201004281239.jpg')

img=cv2.resize(img, (1920,1280))
cv2.imshow('original', img)


lab= cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl = clahe.apply(l)
limg = cv2.merge((cl,a,b))
pimg = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
cv2.imshow('processed', pimg)



k = cv2.waitKey(0)
cv2.destroyAllWindows()