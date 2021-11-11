#!/usr/bin/env python

'''
Multithreaded video processing sample.
Usage:
   video_threaded.py {<video device number>|<video file name>}

   Shows how python threading capabilities can be used
   to organize parallel captured frame processing pipeline
   for smoother playback.

Keyboard shortcuts:

   ESC - exit
   space - switch between multi and single threaded processing
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv

from multiprocessing.pool import ThreadPool
from collections import deque

from player.common import clock, draw_str, StatValue
import player.video as video
import time




# FACE LANDMARKS
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh



def process_frame_default(frame, t):
    return frame, t



class DummyTask:
    def __init__(self, data):
        self.data = data
    def ready(self):
        return True
    def get(self):
        return self.data
    
    

class VideoPlayer:
    
    MIN_SECONDS_LOADED = 5
    PENDING_MAX = 250
    MAX_THREADS_NUMBER = 2
    TIME_AVERAGE_FRAME = 2
    
    
    def __init__(self, video_path):
        # Video data
        self.video_path = video_path
        self.cap = video.create_capture(self.video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Used to process frames
        self.threaded_mode = True
        self.nb_of_threads = min(VideoPlayer.MAX_THREADS_NUMBER, cv.getNumberOfCPUs())
        self.pool = ThreadPool(processes = self.nb_of_threads)
        self.pending = deque()
        self.processing = deque()
        self.process_frame_function = process_frame_default
        
        # Metrics
        self.time_stamps_for_fps = []
        self.actual_fps_count = 0
        self.time_waiting = 0
        self.frame_showed_index = 0
        
        # Used to display the frames
        self.playing = False
        self.draw_function = None
        self.time_last_frame = -1
        
        
        
    def play(self):
        while True:
            
            # On met à jour l'état des frames
            while len(self.processing) > 0 and self.processing[0].ready():
                self.pending.append(self.processing.popleft())
                
            # Si on a chargé une durée suffisante
            if not(self.playing) and len(self.pending) >= VideoPlayer.MIN_SECONDS_LOADED * self.fps:
                self.playing = True
                self.time_last_frame = clock()
            
            # Si la frame est dispo et qu'on a suffisament attendu, on l'affiche
            if len(self.pending) > 0 and self.pending[0].ready():
                if self.playing and (clock() - self.time_last_frame) * self.fps > 1:
                    
                    # Pour mesurer le FPS réel
                    t = clock()
                    self.time_stamps_for_fps.append(t)
                    self.actual_fps_count += 1
                    while len(self.time_stamps_for_fps) > 0 and t - self.time_stamps_for_fps[0] > VideoPlayer.TIME_AVERAGE_FRAME:
                        self.time_stamps_for_fps.pop(0)
                        self.actual_fps_count -= 1
                    
                    # Affichage de l'image avec les infos
                    res, t0 = self.pending.popleft().get()
                    draw_str(res, (20, 20), "threaded      :  " + str(self.threaded_mode))
                    draw_str(res, (20, 40), "FPS            :  %.1f " % (self.actual_fps_count / VideoPlayer.TIME_AVERAGE_FRAME))
                    draw_str(res, (20, 60), "Pending        :  %.1f " % (len(self.pending)))
                    draw_str(res, (20, 80), "Processing     :  %.1f " % (len(self.processing)))
                    cv.imshow('threaded video', res)
                    
                    # On n'utilise pas last_frame = clock() car on risque d'accumuler de l'erreur ce qui baissera le FPS
                    # Si jamais on a vraiment un trop gros retard alors là on utilise clock pour ne pas forcer à affciher
                    # trop de frames d'un coup (ce qui viderait la pending liste)
                    inc_fps = 1./float(self.fps)
                    if clock() - self.time_last_frame < 2*inc_fps:
                        self.time_last_frame += inc_fps
                    else:
                        self.time_last_frame = clock()
            else: 
                # On arrete et on charge un certain nombre de frames
                self.playing = False
                    
                
            if len(self.processing) < self.nb_of_threads and len(self.pending) <= VideoPlayer.PENDING_MAX:
                _ret, frame = self.cap.read()
                t = clock()
                task = None
                if self.threaded_mode:
                    task = self.pool.apply_async(self.process_frame_function, (frame.copy(), t))
                else:
                    task = DummyTask(self.process_frame_function(frame, t))
                self.processing.append(task)
                
            ch = cv.waitKey(1)
            if ch == ord(' '):
                self.threaded_mode = not self.threaded_mode
            if ch == 27:
                break
    
        # Close the window
        cv.destroyAllWindows()
        
"""
def main():
    import sys
    
    with mp_face_mesh.FaceMesh(
    max_num_faces=3,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

        try:
            fn = sys.argv[1]
        except:
            fn = 0
        cap = video.create_capture(r'D:\Google Drive\Polytechnique\3A\INF573\Projet\speechBubbleSubtitles\data\video.mp4')
    
    
        def process_frame(image, t0):
            # some intensive computation...
            if t - int(t) <= 0.05:
                time.sleep(0.5)
        
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)
        
            # Draw the face mesh annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_tesselation_style())
                    
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_contours_style())
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_iris_connections_style())
                
            return image, t0
        
        
   

if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()

"""



