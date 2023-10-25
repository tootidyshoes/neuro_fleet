# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 11:18:12 2019

@author: VANSHIKA
"""

# Drive in remote control mode
# Take the web cam footage
# Receiving steering information including gas paddle (direction)
# Record all in convinient formate

import os
import math
import numpy as np
import glob
import scipy
import scipy.misc
import datetime
import cv2

import argparse

parser = argparse.ArgumentParser(description='Steer autonomours Car')

parser.add_argument('-d','--debug', action = 'store_true', default='Flase')

args = parser.parse_args()
debug = args.debug

import serial
from PIL import Image, ImageDraw
import pygame
import pygame.camera
from pygame.locals import *
from VideoCapture import Device
pygame.init()
pygame.camera.init()

#Initialize the webcam
print('Initializing webcam')
cams = pygame.camera.list_cameras()
cam = pygame.camera.Camera(cams[0],(64,64),'RGB')
cam.start()  

date = datetime.datetime.now()
time_format = '%Y %M  %d %H:%M:%'
imgs_file = 


