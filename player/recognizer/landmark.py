from player.recognizer.position import Position
from player.recognizer.interpolable import Interpolable

import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh



class Landmark:

    MOUTH_MINIMUM_CHANGE_RATIO_TO_DETECT_SPEAKING = 0.25
    MOUTH_NUMBER_OF_FRAMES_TO_DETECT_SPEAKING = 10
    EPSILON = 0.00001 # smal vule used to prevent division by zero
    
    def __init__(self):
        self.values  = []
        self.indexes = []
        self.mouth = Position()
        self.mouth_height = Interpolable()
        self.speaking = Interpolable()
    
    
    def add(self, value, frame_index):
        """Add the position of the landmarks for a given frame identified by frame_index"""
        self.values.append(value)
        self.indexes.append(frame_index)
        self.computeMouthPos(-1)
        
        
    def computeMouthPos(self, value_index):
        """Compute the position of the mouth as the average position of the landmarks for the lips.
        This is computed for the index value_index in self.values"""
        sum_x = 0
        sum_y = 0
        max_y = 0
        min_y = 1
        nb_keypoints = 0
        for key in dict(mp_face_mesh.FACEMESH_LIPS):
            val = dict(mp_face_mesh.FACEMESH_LIPS)[key]
            landmark = self.values[value_index][val]
            sum_x += landmark.x
            sum_y += landmark.y
            max_y = max(max_y, landmark.y)
            min_y = min(min_y, landmark.y)
            nb_keypoints += 1
            
        mouth_x = sum_x / nb_keypoints
        mouth_y = sum_y / nb_keypoints
        self.mouth.add((mouth_x, mouth_y), self.indexes[value_index])
        self.mouth_height.add(max_y - min_y, self.indexes[value_index])

        min_mouth_height = max_y - min_y
        max_mouth_height = max_y - min_y
        val_index = value_index % len(self.indexes)
        while val_index > 0 and (self.indexes[value_index] - self.indexes[val_index-1]) <= Landmark.MOUTH_NUMBER_OF_FRAMES_TO_DETECT_SPEAKING:
            val_index -= 1
            min_mouth_height = min(min_mouth_height, self.mouth_height.get(self.indexes[val_index]))
            max_mouth_height = max(max_mouth_height, self.mouth_height.get(self.indexes[val_index]))

        mouth_ratio = (max_mouth_height - min_mouth_height) / max(max_mouth_height, Landmark.EPSILON) # The max is here to prevent the division by 0
        self.speaking.add((mouth_ratio >= Landmark.MOUTH_MINIMUM_CHANGE_RATIO_TO_DETECT_SPEAKING), self.indexes[value_index])
        
        
    def get(self, frame_index):
        """Returns a dictionnary containing the landmarks and the mouth position for a given frame.
        If this frame was not computed, the landmark returned is the closest previous one. The mouth
        position is interpolated"""
        dict_return = {}
        dict_return["mouth"] = self.mouth.get(frame_index)
        
        # On cherche l'index le plus proche en dessous de frame_index
        value_index = 0
        while (len(self.indexes) > value_index +1) and (self.indexes[value_index+1] <= frame_index):
            value_index += 1
            
        dict_return["landmarks"] = self.values[value_index]
        return dict_return
    
    
    def cleanup(self, frame_index):
        """Delete all the information on the frame previous to frame_index"""
        self.mouth.cleanup(frame_index)
        while  (len(self.indexes) > 0) and self.indexes[0] < frame_index:
             self.indexes.pop(0)
             self.values.pop(0)



    def merge(self, other):
        self.mouth.merge(other.mouth)
        self.indexes.extend(other.indexes)
        self.values.extend(other.values)