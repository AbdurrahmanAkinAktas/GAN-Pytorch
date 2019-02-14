# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 19:48:07 2019

@author: McGurk
"""

import cv2
import os

for image in os.listdir('ppm'):
    img_ppm = cv2.imread(os.path.join('ppm',image))
    # img_rgb = cv2.cvtColor(img_ppm, cv2.COLOR_)
    cv2.imwrite('rgb/'+image.replace('ppm','jpg'), img_ppm)

    