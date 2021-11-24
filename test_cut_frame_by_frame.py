# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 11:28:52 2021

@author: josse
"""

# Standard PySceneDetect imports:
from scenedetect import VideoManager
from scenedetect import SceneManager
import cv2
"""
# For content-aware scene detection:
from scenedetect.detectors import ContentDetector

def find_scenes(video_path, threshold=30.0):
    # Create our video & scene managers, then add the detector.
    video_manager = VideoManager([video_path])
    detector = ContentDetector(threshold=threshold)
    
    frame_index = -1
    video_manager.start()
    cuts = []
    nb_frames = video_manager.get_duration()[0].get_frames()

    while True:
        frame_index += 1
        if frame_index % int(nb_frames/1000.0) == 0:
            progress = 100*frame_index / float(nb_frames)
            print("[" + "=" * int(progress) + " "*(100-int(progress)) + "] " + str(round(progress,2)) + "%", end="\r")
        ret_val, frame = video_manager.read()
        if not ret_val: break
        frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        cuts += detector.process_frame(frame_index, frame)

    return cuts

scenes = find_scenes('data/video.mp4')
print(scenes)
"""

from scenecuts import CutDetector


# For content-aware scene detection:
from scenedetect.detectors import ContentDetector
detector = CutDetector()

def find_scenes(video_path, threshold=30.0):
    # Create our video & scene managers, then add the detector.
    video_manager = VideoManager([video_path])
    
    frame_index = -1
    video_manager.start()
    nb_frames = video_manager.get_duration()[0].get_frames()

    while True:
        frame_index += 1
        if frame_index % int(nb_frames/1000.0) == 0:
            progress = 100*frame_index / float(nb_frames)
            print("[" + "=" * int(progress) + " "*(100-int(progress)) + "] " + str(round(progress,2)) + "%", end="\r")
        ret_val, frame = video_manager.read()
        if not ret_val: break

        frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        detector.process(frame_index, frame)

find_scenes('data/video.mp4')
print(detector.cuts)