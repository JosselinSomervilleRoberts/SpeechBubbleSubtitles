# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 00:04:39 2021

@author: josse
"""

from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
from time import time


def verify(img1_path,img2_path):
    img1= cv2.resize(cv2.imread(img1_path), (0, 0), fx=0.25, fy=0.25)
    img2= cv2.resize(cv2.imread(img2_path), (0, 0), fx=0.25, fy=0.25)
    
    #plt.imshow(img1[:,:,::-1])
    #plt.show()
    plt.imshow(img2[:,:,::-1])
    plt.show()
    t0 = time()
    output = DeepFace.verify(img1_path,img2_path, model_name="Facenet")
    t1 = time()
    print("Time spent :", t1-t0)
    print(output)
    verification = output['verified']
    if verification:
       print('They are same')
    else:
       print('The are not same')
       


verify('data/jake.jpg','data/jake2.JPG')
