from player.recognizer.position import Position

import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh



class Landmark:
    
    def __init__(self):
        self.values  = []
        self.indexes = []
        self.mouth = Position()
    
    
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
        nb_keypoints = 0
        for key in dict(mp_face_mesh.FACEMESH_LIPS):
            val = dict(mp_face_mesh.FACEMESH_LIPS)[key]
            landmark = self.values[value_index][val]
            sum_x += landmark.x
            sum_y += landmark.y
            nb_keypoints += 1
            
        mouth_x = sum_x / nb_keypoints
        mouth_y = sum_y / nb_keypoints
        self.mouth.add((mouth_x, mouth_y), self.indexes[value_index])
        
        
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