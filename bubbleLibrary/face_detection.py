# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 14:28:49 2021

@author: josse
"""


import imutils
import numpy as np
import cv2
from base64 import b64decode
import os


class FaceDetector:
    
    def __init__(self):
        print(os.getcwd())
        prototxt = 'bubbleLibrary/face_model/deploy.prototxt'
        model = 'bubbleLibrary/face_model/res10_300x300_ssd_iter_140000.caffemodel'
        self.net = cv2.dnn.readNetFromCaffe(prototxt, model)


    def detect(self, image):
        #image = imutils.resize(image, width=400)
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()
        return detections
