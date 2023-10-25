# -*- coding: utf-8 -*-
"""
Created on Thu May 30 13:32:43 2019

@author: VANSHIKA
"""
#importing the packages
import cv2
import numpy as np
import os

# Playing video from file:
FPS = 20
cap = cv2.VideoCapture('Intro.mp4')
cap.set(cv2.CAP_PROP_FPS, FPS)

try:
    if not os.path.exists('data'):
        os.makedirs('data')
except OSError:
    print ('Error: Creating directory of data')

currentFrame = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Saves image of the current frame in jpg file
    name = './data/frame' + str(currentFrame) + '.jpg'
    print ('Creating...' + name)
    cv2.imwrite(name, frame)

    # To stop duplicate images
    currentFrame += 1
    
    if not ret: break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

