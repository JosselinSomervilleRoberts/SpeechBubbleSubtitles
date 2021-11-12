from recognizer.position import Position
from recognizer.interpolable import Interpolable

import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh



class Landmark:

    MOUTH_MINIMUM_CHANGE_RATIO_TO_DETECT_SPEAKING = 0.22
    MOUTH_NUMBER_OF_FRAMES_TO_DETECT_SPEAKING = 10
    EPSILON = 0.00001 # smal vule used to prevent division by zero
    MINIMUM_NUMBER_OF_FRAMES_NOT_SPEAKING = 10
    MINIMUM_NUMBER_OF_FRAMES_SPEAKING = 20
    
    def __init__(self):
        self.values  = []
        self.indexes = []
        self.mouth = Position()
        self.mouth_height = Interpolable()

        # For speaking detection
        self.speaking = Interpolable()
        self.first_spoke = -100
        self.last_spoke = -100
    
    
    def add(self, value, frame_index):
        """Add the position of the landmarks for a given frame identified by frame_index"""
        self.values.append(value)
        self.indexes.append(frame_index)
        self.computeMouthPos(-1)
        
        
    def computeMouthPos(self, value_index):
        """Compute the position of the mouth as the average position of the landmarks for the lips.
        This is computed for the index value_index in self.values"""
        value_index = value_index % len(self.indexes)

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
        val_index = value_index 
        while val_index > 0 and (self.indexes[value_index] - self.indexes[val_index-1]) <= Landmark.MOUTH_NUMBER_OF_FRAMES_TO_DETECT_SPEAKING:
            val_index -= 1
            min_mouth_height = min(min_mouth_height, self.mouth_height.get(self.indexes[val_index]))
            max_mouth_height = max(max_mouth_height, self.mouth_height.get(self.indexes[val_index]))

        mouth_ratio = (max_mouth_height - min_mouth_height) / max(max_mouth_height, Landmark.EPSILON) # The max is here to prevent the division by 0
        speaking = float(mouth_ratio >= Landmark.MOUTH_MINIMUM_CHANGE_RATIO_TO_DETECT_SPEAKING)
        self.speaking.add(speaking, self.indexes[value_index])


        #if value_index>0:
        #    print(speaking, self.last_spoke == self.indexes[value_index-1])
        if value_index>0 and speaking and self.last_spoke < self.indexes[value_index-1]: # Si on parle alors qu avant on parlait pas

            # Le personnage n'a pas encore parlé mais on vient de le créer
            if self.last_spoke < 0 and (self.indexes[value_index] - self.indexes[0]) < Landmark.MINIMUM_NUMBER_OF_FRAMES_NOT_SPEAKING -1:
                for i in range(value_index):
                    self.speaking.values[i] = True
                self.first_spoke = self.indexes[0]

            # Le personnage a arreter de parler pour un temps trop court, on considere qu'il n a pas arreté de parler
            elif 1 + self.indexes[value_index] - self.last_spoke < Landmark.MINIMUM_NUMBER_OF_FRAMES_NOT_SPEAKING:
                val_index = value_index - 1
                while val_index > 0 and self.indexes[val_index] > self.last_spoke:
                    self.speaking.values[val_index] = True
                    val_index -= 1
        
        # Si on ne parle pas depuis longtemps et qu'on a parlé avant, on vérifie pour combien de temps on a parlé, c'était peut etre une erreur
        if not(speaking) and self.first_spoke > 0 and self.indexes[val_index] - self.last_spoke > Landmark.MINIMUM_NUMBER_OF_FRAMES_NOT_SPEAKING:
            # Si on a pas parlé longtemps
            if self.last_spoke - self.first_spoke < Landmark.MINIMUM_NUMBER_OF_FRAMES_SPEAKING:
                # On modifie tous les speaking
                val_index = value_index - 1
                while val_index > 0 and self.indexes[val_index] >= self.first_spoke:
                    self.speaking.values[val_index] = False
                    val_index -= 1

                # On a pas parlé
                self.last_spoke = - 1 - max(Landmark.MINIMUM_NUMBER_OF_FRAMES_SPEAKING, Landmark.MINIMUM_NUMBER_OF_FRAMES_NOT_SPEAKING)
                self.first_spoke = self.last_spoke
        
        if (speaking):
            self.last_spoke = self.indexes[value_index]
            if self.first_spoke < 0: self.first_spoke = self.last_spoke

        
        
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
        self.speaking.merge(other.speaking)