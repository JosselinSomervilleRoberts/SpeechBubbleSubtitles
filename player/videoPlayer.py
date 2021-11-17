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



import numpy as np
import cv2 as cv
from multiprocessing.pool import ThreadPool
from collections import deque

from player.common import clock, draw_str
import player.video as video






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
    MAX_THREADS_NUMBER = 1
    TIME_AVERAGE_FRAME = 2
    
    
    def __init__(self, video_path):
        # Video data
        self.video_path = video_path
        self.cap = video.create_capture(self.video_path)
        self.cap_display = video.create_capture(self.video_path)
        self.current_frame_index = -1
        self.fps = self.cap.get(cv.CAP_PROP_FPS)
        
        # Used to process frames
        self.threaded_mode = True
        self.nb_of_threads = min(VideoPlayer.MAX_THREADS_NUMBER, cv.getNumberOfCPUs())
        self.pool = ThreadPool(processes = self.nb_of_threads)
        self.pending = deque()
        self.processing = deque()
        
        # Metrics
        self.time_stamps_for_fps = []
        self.actual_fps_count = 0
        self.time_waiting = 0
        
        # Used to display the frames
        self.playing = False
        self.draw_function = None
        self.time_last_frame = -1
        
        
        
    def process_frame(self, frame, t):
        return frame, t


    def display(self):
        """Function used to render the frame, adding useful metrics"""
        draw_str(self.current_frame, (20, 20), "threaded      :  " + str(self.threaded_mode))
        draw_str(self.current_frame, (20, 40), "FPS            :  %.1f " % (self.actual_fps_count / VideoPlayer.TIME_AVERAGE_FRAME))
        draw_str(self.current_frame, (20, 60), "Pending        :  %.1f " % (len(self.pending)))
        draw_str(self.current_frame, (20, 80), "Processing     :  %.1f " % (len(self.processing)))
        cv.imshow('threaded video', self.current_frame)
        

    def prepare_display(self):
        """This function replaces the variable self.current_frame using the video for display (self.cap_display)
        and the information processed (self.pending[0])"""
        _ret, self.current_frame = self.cap_display.read()
        self.current_frame_index += 1
        self.pending.popleft()
        
        
    def play(self):
        frame_index_processed = 0
        
        while True:
            
            #print(len(self.pending), "/", len(self.processing))
            
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
                    self.prepare_display()
                    self.display()
                    
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
                task = None
                if self.threaded_mode:
                    task = self.pool.apply_async(self.process_frame, (frame.copy(), frame_index_processed))
                else:
                    task = DummyTask(self.process_frame(frame, frame_index_processed))
                self.processing.append(task)
                frame_index_processed += 1
                
            ch = cv.waitKey(1)
            if ch == ord(' '):
                self.threaded_mode = not self.threaded_mode
            if ch == 27:
                break
    
        # Close the window
        cv.destroyAllWindows()