# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 14:00:00 2021

@author: josse
"""

# pip install ass
# pip install numpy
# pip install cv2
import ass
import numpy as np
import cv2


def convert_time_delta_to_frame_index(time_delta, fps=30):
    return int(time_delta.total_seconds() * fps)


def read_subtitles(file_sub_name, fps=30):
    subtitles = []
    
    if file_sub_name[-4:] == ".ass":     
        with open(file_sub_name, encoding='utf_8_sig') as f:
            file_sub = ass.parse(f)
            for sub in file_sub.events:
                if sub.text[:4].lower() != "sync": # To remove the last message
                    subtitles.append({"start": convert_time_delta_to_frame_index(sub.start, fps),
                                      "end":   convert_time_delta_to_frame_index(sub.end,   fps),
                                      "name":  sub.name,
                                      "text":  sub.text})
    else:
        raise Exception("Subtitle format not supported", file_sub_name[file_sub_name.find("."):])
    
    return subtitles



def read_video(file_video_name):
    cap = cv2.VideoCapture(file_video_name)
    return cap


def read_both(file_video_name, file_sub_name):
    video = read_video(file_video_name)
    fps = video.get(cv2.CAP_PROP_FPS)
    subtitles = read_subtitles(file_sub_name, fps)
    return {"video": video,
            "subtitles": subtitles,
            "fps": fps}