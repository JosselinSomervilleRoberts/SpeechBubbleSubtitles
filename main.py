# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 13:33:59 2021

@author: josse
"""

from bubbleLibrary import read_both
from bubbleLibrary import FaceDetector
from bubbleLibrary import draw_bubble_text


data = read_both("data/video.mp4", "data/video.ass")
print(data)

"""
# success, image = data["video"].read()
cap = data["video"]
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:

    # Display the resulting frame
    cv2.imshow('Frame',frame)

    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

  # Break the loop
  else: 
    break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
"""

import cv2
import numpy as np
import imutils

#ffpyplayer for playing audio
from ffpyplayer.player import MediaPlayer
video_path = "data/video.mp4"
detector = FaceDetector()


def PlayVideo(video_path):
    video = data["video"]
    fps = data["fps"]
    player = MediaPlayer("data/video.mp4")
    sleep_ms = int(np.round((1/fps)*1000))
    frame_count = 0
    index_subtitles = 0
    center = (400, 100)
    bubble_width = 500
    bubble_height = 100
    new_w = 1500
    new_h = 900
    
    while True:
        grabbed, frame=video.read()
        audio_frame, val = player.get_frame()
        if not grabbed:
            print("End of video")
            break
        if cv2.waitKey(28) & 0xFF == ord("q"):
            break
        #cv2.rectangle(frame, (0, 0), (100, 100), (255,0,0), 2)

        image = imutils.resize(frame, width=400)
        (h, w) = image.shape[:2]
        detections = detector.detect(image)
        i_max = -1
        conf_max = 0
        for i in range(0, detections.shape[2]):

        	# extract the confidence (i.e., probability) associated with the prediction
        	confidence = detections[0, 0, i, 2]
        	if confidence > conf_max:
        		conf_max = confidence
        		i_max = i
                
            
        
        	# filter out weak detections by ensuring the `confidence` is
        	# greater than the minimum confidence threshold
        	if confidence > 0.5:
        		# compute the (x, y)-coordinates of the bounding box for the object
        		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        		(startX, startY, endX, endY) = box.astype("int")
        		# draw the bounding box of the face along with the associated probability
        		text = "{:.2f}%".format(confidence * 100)
        		y = startY - 10 if startY - 10 > 10 else startY + 10
        		cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
        		cv2.putText(image, text, (startX, y),
        			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        
        (face_img_h, face_img_w) = image.shape[:2]
        frame = cv2.resize(image, (new_w, new_h))
        
        if frame_count > data["subtitles"][index_subtitles]["end"]:
            index_subtitles += 1
            
        
        if frame_count >= data["subtitles"][index_subtitles]["start"]:
            if i_max >= 0:
                box = detections[0, 0, i_max, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                draw_bubble_text(frame, center, bubble_width, bubble_height, (int(new_w*startX/face_img_w), int(new_h*(startY +int( 0.7*(endY - startY)))/face_img_h)))
            cv2.putText(frame, data["subtitles"][index_subtitles]["text"], (int(center[0]-0.4*bubble_width), int(center[1] + 0.25 * bubble_height)),
        			cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
        frame_count += 1

        cv2.imshow("Video", frame)
        if val != 'eof' and audio_frame is not None:
            #audio
            img, t = audio_frame
    video.release()
    cv2.destroyAllWindows()
    
PlayVideo(video_path)