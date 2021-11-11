from player.videoPlayer import VideoPlayer
from player.recognizer.face import Face
from bubbleLibrary import FacesDetector

from math import sqrt
import numpy as np
from numpy import unravel_index


# FACE LANDMARKS
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh










import face_recognition


# Faces detector
detector = FacesDetector()
#detector.addKnownFace("data/josselin.jpg", "Josselin")
#detector.addKnownFace("data/jake.jpg", "jake")
detector.addKnownFace("data/jake2.JPG", "jake")
dist_max_recognition = 0.55 # Distance below which we assume the recognition is correct


def getBoxFromLandmark(landmark, frame_width, frame_height):
    cx_min=  frame_width
    cy_min = frame_height
    cx_max= cy_max= 0
        
        
    for id, lm in enumerate(landmark):
        cx, cy = int(lm.x * frame_width), int(lm.y * frame_height)

        if cx < cx_min:
            cx_min = cx
        if cy < cy_min:
            cy_min = cy
        if cx > cx_max:
            cx_max = cx
        if cy > cy_max:
            cy_max = cy
                
    # From top-left/bottom-right ------> To Center/width-height
    w = cx_max - cx_min
    h = cy_max - cy_min
    x = int((cx_min + cx_max) / 2.)
    y = int((cy_min + cy_max) / 2.)
    return (x,y,w,h)



def recognize(frame, box, face, f_index):
    # Find who
    if True:
        print("RECOGNIZE")
        enlarge_coef = 1.6
        new_cx_min = max(0, int(box[0] - 0.5 * enlarge_coef * box[2]))
        new_cy_min = max(0, int(box[1] - 0.5 * enlarge_coef * box[3]))
        new_cx_max = int(box[0] + 0.5 * enlarge_coef * box[2])
        new_cy_max = int(box[1] + 0.5 * enlarge_coef * box[3])
        cropped = frame[new_cy_min:new_cy_max, new_cx_min:new_cx_max]
        small_frame = cv2.resize(cropped, (0, 0), fx=0.5, fy=0.5)
        
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_small_frame)
        list_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations, num_jitters=1, model="small")
        
        if len(list_encodings) > 0:
            unknown_face_encoding = list_encodings[0]
            distances = face_recognition.face_distance(detector.known_face_encodings, unknown_face_encoding)
            
            index_min = np.argmin(distances)
            min_distance = distances[index_min]
                    
            if min_distance < dist_max_recognition:
                face.name = detector.known_face_names[index_min]
    

















class VideoPlayerWithBubbles(VideoPlayer):
    
    SIMILIRATY_THRESHOLD = 0.4
    REFRESH_RECOGNITION = 10
    THRESHOLD_TIMES_MISSED_ALLOWED_MAX = 3
    
    def __init__(self, video_path, subtitle_path, pool):
        VideoPlayer.__init__(self, video_path, pool)
        
        # Video data
        self.subtitle_path = subtitle_path
        
        # Display option
        self.show_mesh = False
        self.show_outline = True
        
        # General frame infos
        self.frame_width = None
        self.frame_height = None
        self.frame_normalizer = None
        
        # Face mesh and landmarks
        self.faces = []
        self.face_mesh = mp_face_mesh.FaceMesh(
                            max_num_faces=3,
                            refine_landmarks=True,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.5)
        
        
        
    def process_frame_old(self, frame, t):
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        frame.flags.writeable = False
        frame= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame)
        
        # Draw the face mesh annotations on the image.
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                
                if self.show_mesh:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_tesselation_style())
                
                if self.show_outline:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_contours_style())
                
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_iris_connections_style())
                    
                
        return frame, t
    
    
    def process_frame(self, frame, frame_index):
        
        print("faces =", len(self.faces))
        
        # We have never loaded the general frame infos
        if self.frame_width is None:
            self.frame_width = frame.shape[1]
            self.frame_height = frame.shape[0]
            self.frame_normalizer = sqrt(self.frame_width**2 + self.frame_height**2) # To have distance between 0 and 1
            Face.frame_width  = self.frame_width
            Face.frame_height = self.frame_height
            Face.frame_normalizer = self.frame_normalizer
    
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        frame.flags.writeable = False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # We perform face frame detection
        results = self.face_mesh.process(frame)
    
        # We give the image its color back
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        frame.flags.writeable = False
        
        
        # If we detect at least one head
        if results.multi_face_landmarks:
            
            landmarks = [face_landmark.landmark for face_landmark in results.multi_face_landmarks] # Landmarks of the heads
            boxes = [getBoxFromLandmark(landmark, self.frame_width, self.frame_height) for landmark in landmarks] # Bounding boxes of heads
            available_faces_indexes = list(range(len(self.faces))) # Previous faces available for matching
            
            # Here we assign detected faces to already existing faces
            continuer = True
            while continuer and (len(available_faces_indexes) > 0) and (len(boxes) > 0):
                sim_mat = np.array([self.faces[face_index].similarity(frame_index, boxes) for face_index in available_faces_indexes])

                # S'il y a encore un matching possible, cad au moins 1 ancienne tete non assignée
                # et au moins 1 tete detectée non assignée
                if len(sim_mat.shape) > 1:
                    arg_max = np.argmax(sim_mat)
                    # tupple (index in existing faces object, index in detected face)
                    index_max = unravel_index(arg_max, sim_mat.shape)
                    existing_index = available_faces_indexes[index_max[0]]
                    detected_index = index_max[1]
                    
                    # If the similarity is enough, we assign the detection to the existing face
                    if sim_mat[index_max] > VideoPlayerWithBubbles.SIMILIRATY_THRESHOLD:
                        
                        # if the face is unidentified we try to recognize it every so often
                        if ((frame_index + index_max[0]) % VideoPlayerWithBubbles.REFRESH_RECOGNITION == 0) and self.faces[existing_index].name is None:
                            recognize(frame, boxes[detected_index], self.faces[existing_index], frame_index)
                            
                            
                        self.faces[existing_index].detected(boxes[detected_index],
                                                     landmarks[detected_index],
                                                     frame_index)
                        boxes.pop(detected_index)
                        landmarks.pop(detected_index)
                        available_faces_indexes.pop(index_max[0])
                    else:
                        continuer = False
                else:
                    continuer = False
        
    
            # This boxes are unassigned
            # This means that a mesh was detected but no exiting face was found to match it
            # Therefore for each of these boxes, we must create a new face
            for index, box in enumerate(boxes):
                new_face = Face()
                new_face.detected(box, landmarks[index], frame_index)    
                #recognize(frame, box, new_face, frame_index)
                self.faces.append(new_face)
                
            
            """
            # We loop through the existing faces to tag the faces that did not get detected
            index = 0
            while index < len(self.faces):
                face = self.faces[index]
                incrementIndex = True
                if face.last_frame_update < frame_index:
                    face.nb_times_missed += 1
                    
                    if face.nb_times_missed >= VideoPlayerWithBubbles.THRESHOLD_TIMES_MISSED_ALLOWED_MAX:
                        self.faces.pop(index)
                        incrementIndex = False
                        
                if incrementIndex: index += 1
            """
    
            for face in self.faces:
                face.draw(frame, frame_index)
                
        return frame, frame_index
