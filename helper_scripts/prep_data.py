# -*- coding: utf-8 -*-

import cv2
import os

for image in os.listdir('rgb'):
    
    img = cv2.imread(os.path.join('rgb',image))
    width = img.shape[1]
    height= img.shape[0]
        
    img=cv2.resize(img, (400,256))
    img = img[0:256,0:256]


    lab= cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    
    proc_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    
    
    
    cv2.imwrite('rgb_clahe_resized_256x256/'+image, proc_img)
    

    
