# -*- coding: utf-8 -*-
"""
Author: Andreas Rössler
"""
import os
from os.path import join
from tqdm import tqdm
import cv2
import string
import random



def extract_sequences(video_path):
    print("extract_sequences")
    vidcap = cv2.VideoCapture(video_path)
    file_name = os.path.splitext(os.path.basename(video_path))[0]
    file_dir = os.path.dirname(video_path )
    if os.path.exists(join(file_dir, file_name)):
        file_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
    os.makedirs(join(file_dir, file_name))
    
    
    sec = 0
    frameRate = 0.1 #//it will capture image in each 0.5 second
    count=0
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    success,image = vidcap.read()
    
    while success:
        file = '{:04}'.format(count)
        print(file)
        file = "%s.png" % file
        file = join(file_dir, file_name, file)
        
        print(file)
        sec = sec + frameRate
        sec = round(sec, 2)
        vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
        success,image = vidcap.read()
        if success:
            cv2.imwrite(file, image)     # save frame as JPG file
            count = count + 1
        if count == 100:
            break
        
    return join(file_dir, file_name), count+1

