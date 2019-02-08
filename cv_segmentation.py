import os
import numpy as np
import cv2
import csv


def find_parts(skeleton_reader):

    for row in skeleton_reader:
        head_color = (float(row['Head_color_X'].replace(',', '.')), float(row['Head_color_Y'].replace(',', '.')))
        head_depth = (float(row['Head_depth_X'].replace(',', '.')), float(row['Head_depth_Y'].replace(',', '.')))

        hand_left_color = (float(row['WristLeft_color_X'].replace(',', '.')), float(row['WristLeft_color_Y'].replace(',', '.')))
        hand_left_depth = (float(row['WristLeft_depth_X'].replace(',', '.')), float(row['WristLeft_depth_Y'].replace(',', '.')))

        hand_right_color = (float(row['WristRight_color_X'].replace(',', '.')), float(row['WristRight_color_Y'].replace(',', '.')))
        hand_right_depth = (float(row['WristRight_depth_X'].replace(',', '.')), float(row['WristRight_depth_Y'].replace(',', '.')))



    return (head_color, head_depth, hand_left_color, hand_left_depth, hand_right_color, hand_right_depth)


def mark_parts(path, parts):
    img = cv2.imread(path)

    for part in parts:
        img = cv2.circle(img, part, 5, (255,0,0), 2)





skeleton_file = open(, encoding='utf8')
skeleton_reader = csv.DictReader(skeleton_file)
