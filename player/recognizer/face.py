import numpy as np
import cv2
from bubbleLibrary.utils_cv2 import dist
from player.recognizer.position import Position
from player.recognizer.interpolable import Interpolable
from player.recognizer.landmark import Landmark

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh



class Face:

    global_id_counter = 0

    frame_width  = 0
    frame_height = 0
    frame_normalizer = 0

    NOT_APPEARED_MAX_NUMBER_FRAMES = 10
    MINIMUM_APPEARED_NUMBER_FRAMES = 10
    
    def __init__(self):
        
        # Character info
        self.id = Face.global_id_counter
        Face.global_id_counter += 1
        self.name = None             # String: name of the character
        self.first_appearance = None
        self.last_appearance  = None
        
        # Position
        self.center = Position()   # x coordinate of the center of the head bounding box
        self.w = Interpolable()   # width  of the head bounding box
        self.h = Interpolable()   # height of the head bounding box
        self.landmarks = Landmark()
        


    def isPresent(self, frame_index):
        return (self.first_appearance <= frame_index <= self.last_appearance)
        

        
    def detected(self, box, landmarks, frame_index):
        # Note that we recognised this character
        if self.first_appearance is None: self.first_appearance = frame_index
        self.last_appearance = frame_index
        
        # Update postion
        self.center.add( (box[0], box[1]), frame_index )
        self.w.add( box[2] , frame_index )
        self.h.add( box[3] , frame_index )
        
        # Update landmark
        self.landmarks.add(landmarks, frame_index)
        
        
        
    def similarity(self, frame_index, boxes):
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
        
        expected_pos  = self.center.get(frame_index)
        expected_area =  self.w.get(frame_index) * self.h.get(frame_index)
        
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
            #print("area_ratio =", area_ratio)
            #print("area =", area)
            #print("expected_area =", expected_area)
            
            
            # Movement of the head
            distance = dist(pos, expected_pos) / Face.frame_normalizer
            #print("distance =", distance)
            #print("pos =", pos)
            #print("expected_pos =", expected_pos)
            #print("norm_factor =", norm_factor)
            
            # Final score
            score = area_ratio * (1 - distance**0.4)
            proximity_scores[index] = score
            
        return proximity_scores
        
        
        
    def getMouthPos(self, frame_index):
        mouth_relative = self.landmarks.get(frame_index)["mouth"]
        mouth_x = int(Face.frame_width  * mouth_relative[0])
        mouth_y = int(Face.frame_height * mouth_relative[1])
        return (mouth_x, mouth_y)
    
    
    
    def __eq__(self, other):
        """
        To check if two faces are equal we simply compare their name
        """
        if other is None:
            return False # if the other face is unidentified then they are not equal
        elif isinstance(other, Face): # We compare two Faces
            if self.name is None or other.name is None: return False # If one of the face is unidentified, then the faces are not equal
            return self.name.lower() == other.name.lower() 
        elif isinstance(other, str):
            return self.name.lower() == other.lower()
        
        return False
    
    
    
    def draw(self, frame, frame_index):
        center = self.getMouthPos(frame_index)
        axes = (10,5)
        angle = 360
        text = "Unknown"
        if not(self.name is None): text = self.name
        color = (255, 0, 0)
        if self.landmarks.speaking.get(frame_index, average_nb = 21) > 0.15: color = (0,0,255)
        
        # Draw
        cv2.ellipse(frame, center, axes, angle, 0 , 360, color, 2)
        cv2.putText(frame, text, (-len(text) * 5 + center[0], 20 + center[1]),
        			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)


    
    def merge(self, other_face):
        if self.name is None: self.name = other_face.name
        self.last_appearance = other_face.last_appearance

        # Merging positions array
        self.center.merge(other_face.center)
        self.w.merge(other_face.w)
        self.h.merge(other_face.h)
        self.landmarks.merge(other_face.landmarks)

        
    def getLastKnownPos(self):
        return self.center.get(self.last_appearance)


    def getFirstKnownPos(self):
        return self.center.get(self.first_appearance)

    
    def checkState(self, frame_index):
        """Returns if the face is still active and if it should be destroyed (it was an outlier)"""

        # Check if the face is still alive
        if self.last_appearance is None: return False, True
        if frame_index - self.last_appearance > Face.NOT_APPEARED_MAX_NUMBER_FRAMES:
            # The face has not appeared for too many frames

            # If the face was only detected for a very short time period, then it must be an error in the detection
            if self.last_appearance - self.first_appearance < Face.MINIMUM_APPEARED_NUMBER_FRAMES:
                return False, True
            
            # The face is no longer active but was present long enough so the detection must be right
            return False, False

        # The face is active, we keep it
        return True, False

    
    def cleanup(self, frame_index):
        """Cleanup data no longer used to optimize the performance"""
        self.center.cleanup(frame_index)
        self.w.cleanup(frame_index)
        self.h.cleanup(frame_index)
        self.landmarks.cleanup(frame_index)
