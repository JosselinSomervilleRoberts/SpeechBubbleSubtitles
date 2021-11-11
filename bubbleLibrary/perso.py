# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 11:52:11 2021

@author: josse
"""

import numpy as np
import cv2
from bubbleLibrary.utils_cv2 import dist
from bubbleLibrary.integrableNumber import IntegrableNumber

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


class Perso:
    frame_width  = 0
    frame_height = 0
    
    def __init__(self):
        
        # Character info
        self.name = "Unknown"                # String: name of the character
        #self.ref_image = None         # Image for face recognition
        self.nb_times_missed = 0     # Number of times in a row, this character was not found via recognition
        
        # Position
        self.last_frame_update = -1   # Index of the last frame the face was detected
        self.x = IntegrableNumber()   # x coordinate of the center of the head bounding box
        self.y = IntegrableNumber()   # y coordinate of the center of the head bounding box
        self.w = IntegrableNumber()   # width  of the head bounding box
        self.h = IntegrableNumber()   # height of the head bounding box
        self.mouth_x = IntegrableNumber()   # x coordinate of the center of the mouth
        self.mouth_y = IntegrableNumber()   # y coordinate of the center of the mouth
        self.landmarks = None
        
        
        
    def detected(self, box, landmarks, frame_index):
        # Note that we recognised this character
        self.nb_times_missed = 0
        self.last_frame_update = frame_index
        
        # Update postion
        self.x.update( box[0] , frame_index )
        self.y.update( box[1] , frame_index )
        self.w.update( box[2] , frame_index )
        self.h.update( box[3] , frame_index )
        
        self.landmarks = landmarks
        self.updateMouthPos(frame_index)
        
                         
        
    def updateMouthPos(self, frame_index):
        sum_x = 0
        sum_y = 0
        nb_keypoints = 0
        for key in dict(mp_face_mesh.FACEMESH_LIPS):
            val = dict(mp_face_mesh.FACEMESH_LIPS)[key]
            landmark = self.landmarks[val]
            sum_x += landmark.x
            sum_y += landmark.y
            nb_keypoints += 1
            
        self.mouth_x.update( int(Perso.frame_width  * sum_x / nb_keypoints) , frame_index )
        self.mouth_y.update( int(Perso.frame_height * sum_y / nb_keypoints) , frame_index )

        
        
    def similarity(self, frame_index, boxes, norm_factor=1):
        """
        Update the positions of the face of a character.
        This only works if there are no cuts between the last scene computed and the current one,
        and if the characters did not move a lot between the frames.
        
        Parameters
        ----------
        frame_index : (int) Index of the frame
        boxes       : list of tuple of bounding boxes for faces
        landmarks   : list of landmarks (list of positions)
        norm_factor : (int) sqrt(width**2 + height**2) of the image to have distances between 0 and 1

        Returns
        -------
        None.
        """
        
        # Metrics for the previous box
        #expected_pos  = (self.x.predict(frame_index) , self.y.predict(frame_index))
        #expected_area =  self.w.predict(frame_index) * self.h.predict(frame_index)
        expected_pos  = (self.x.x[0] , self.y.x[0])
        expected_area =  self.w.x[0] * self.h.x[0]
        
        # Find the corresponding position by proximity
        proximity_scores = np.zeros(len(boxes))
        for index, box in enumerate(boxes):
            
            # Metrics for the box
            (center_x, center_y, w, h) = box
            pos = (center_x, center_y)
            area = w * h
            
            # Compute the ratio of area change (between 0 and 1)
            # The max(1, ...) is a way to avoid dividing by zero
            area_ratio = min(max(1, area) / max(1, expected_area), max(1, expected_area) / max(1, area)) 
            
            # Movement of the head
            distance = dist(pos, expected_pos) / norm_factor
            
            # Final score
            score = area_ratio * (1 - distance**0.3)
            proximity_scores[index] = score
            
        return proximity_scores
        
        
        
    def get_mouth_pos(self, frame_index):
        return (self.mouth_x.predict(frame_index), self.mouth_y.predict(frame_index))
    
    
    
    def __eq__(self, other):
        """
        To check if two characters are equal we simply compare their name
        """
        if isinstance(other, Perso):
            return self.name.lower() == other.name.lower() 
        elif isinstance(other, str):
            return self.name.lower() == other.lower()
        
        return False
    
    
    
    def draw(self, frame, frame_index):
        center = (int(self.mouth_x.predict(frame_index)), int(self.mouth_y.predict(frame_index)))
        axes = (10,5)
        angle = 360
        cv2.ellipse(frame, center, axes, angle, 0 , 360, (255,0,0), 2)
        
        # setup text
        cv2.putText(frame, self.name, (-len(self.name) * 5 + int(self.mouth_x.predict(frame_index)), 20 + int(self.mouth_y.predict(frame_index))),
        			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)
        