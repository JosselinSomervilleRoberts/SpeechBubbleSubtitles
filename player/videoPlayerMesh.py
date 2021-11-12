from player import VideoPlayer

from math import sqrt
import numpy as np
from collections import deque
import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh





class VideoPlayerMesh(VideoPlayer):

    def __init__(self, video_path):
        VideoPlayer.__init__(self, video_path)
        
        # Display option
        self.show_mesh = False
        self.show_outline = True

        # Face mesh and landmarks
        self.meshes = deque()
        self.face_mesh = mp_face_mesh.FaceMesh(
                            max_num_faces=3,
                            refine_landmarks=True,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.5)

        
        
    def process_frame(self, frame, t):
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        frame.flags.writeable = False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame)
        self.meshes.append(results)
    
    


    def prepare_display(self):
        """This function replaces the variable self.current_frame using the video for display (self.cap_display)
        and the information processed (self.pending[0])"""

        # Read the frame from video file
        VideoPlayer.prepare_display(self)

        results = self.meshes.popleft()
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                
                if self.show_mesh:
                    mp_drawing.draw_landmarks(
                        image=self.current_frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_tesselation_style())
                
                if self.show_outline:
                    mp_drawing.draw_landmarks(
                        image=self.current_frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_contours_style())
                
                    mp_drawing.draw_landmarks(
                        image=self.current_frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_iris_connections_style())
