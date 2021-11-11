from bubbleLibrary import read_both
from bubbleLibrary import FacesDetector
from bubbleLibrary import draw_bubble_text
from bubbleLibrary.perso import Perso

import cv2
import numpy as np
from numpy import unravel_index
import imutils
from ffpyplayer.player import MediaPlayer
import face_recognition
from math import sqrt
import time



# Read data
video_path = "data/video.mp4"
subtitles_path = "data/video.ass"
data = read_both(video_path, subtitles_path)


# Faces detector
detector = FacesDetector()
#detector.addKnownFace("data/josselin.jpg", "Josselin")
#detector.addKnownFace("data/jake.jpg", "jake")
detector.addKnownFace("data/jake2.jpg", "jake")



import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
faces = []


# PARAMETERS
similarity_threshold = 0.5    # minimum score to accept tracking
nb_times_missed_threshold = 8 # Nb of times we can miss a face before deleting it
dist_max_recognition = 0.55 # Distance below which we assume the recognition is correct
refresh_recognition = 10 # Evevy x frames, we try face_recognition again on this character


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



def recognize(frame, box, face):
    # Find who
    enlarge_coef = 1.6
    new_cx_min = max(0, int(box[0] - 0.5 * enlarge_coef * box[2]))
    new_cy_min = max(0, int(box[1] - 0.5 * enlarge_coef * box[3]))
    new_cx_max = int(box[0] + 0.5 * enlarge_coef * box[2])
    new_cy_max = int(box[1] + 0.5 * enlarge_coef * box[3])
    cropped = frame[new_cy_min:new_cy_max, new_cx_min:new_cx_max]
    small_frame = cv2.resize(cropped, (0, 0), fx=0.5, fy=0.5)
    
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]
    t0 = time.time()
    face_locations = face_recognition.face_locations(rgb_small_frame)
    t1 = time.time()
    print("Locations:", 1000*(t1-t0), "ms")
    list_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations, num_jitters=1, model="small")
    print("Embedding:", 1000*(time.time()-t0), "ms")
    
    if len(list_encodings) > 0:
        unknown_face_encoding = list_encodings[0]
        distances = face_recognition.face_distance(detector.known_face_encodings, unknown_face_encoding)
        
        index_min = np.argmin(distances)
        min_distance = distances[index_min]
                
        if min_distance < dist_max_recognition:
            face.name = detector.known_face_names[index_min]


drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('data/video.mp4')

with mp_face_mesh.FaceMesh(
    max_num_faces=3,
    refine_landmarks=True,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.5) as face_mesh:
    
  frame_index = -1
  while cap.isOpened():
    print("\n")
    t0 = time.time()
    
    # Read the frame
    success, image = cap.read()
    frame_index += 1
    if not success:
      print("Corrupted frame")
      break

    frame_width = image.shape[1]
    frame_height = image.shape[0]
    Perso.frame_width = frame_width
    Perso.frame_height = frame_height
    size_normalizer = sqrt(frame_width**2 + frame_height**2) # To have distance between 0 and 1

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    print("Before mesh:", 1000*(time.time()-t0), "ms")
    results = face_mesh.process(image)
    print("After frame:", 1000*(time.time()-t0), "ms")

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    
    # If we detect at least one head
    if results.multi_face_landmarks:
        
        print("Before arrays:", 1000*(time.time()-t0), "ms")
        landmarks = [face_landmark.landmark for face_landmark in results.multi_face_landmarks] # Landmarks of the heads
        boxes = [getBoxFromLandmark(landmark, frame_width, frame_height) for landmark in landmarks] # Bounding boxes of heads
        available_faces_indexes = list(range(len(faces))) # Previous faces available for matching
        print("After arrays:", 1000*(time.time()-t0), "ms")
        
        # Here we assign detected faces to already existing faces
        continuer = True
        while continuer and (len(available_faces_indexes) > 0) and (len(boxes) > 0):
            sim_mat = np.array([faces[face_index].similarity(frame_index, boxes, norm_factor=size_normalizer) for face_index in available_faces_indexes])
            print("sima_mt", sim_mat)
            
            # S'il y a encore un matching possible, cad au moins 1 ancienne tete non assignée
            # et au moins 1 tete detectée non assignée
            if len(sim_mat.shape) > 1:
                arg_max = np.argmax(sim_mat)
                # tupple (index in existing faces object, index in detected face)
                index_max = unravel_index(arg_max, sim_mat.shape)
                existing_index = available_faces_indexes[index_max[0]]
                detected_index = index_max[1]
                
                # If the similarity is enough, we assign the detection to the existing face
                if sim_mat[index_max] > similarity_threshold:
                    print("- Detected", index_max[1], "assigned to", index_max[0])
                    
                    if ((frame_index + index_max[0]) % refresh_recognition == 0) and faces[existing_index].name.lower() == "unknown":
                        

                        print("Cefore recognition:", 1000*(time.time()-t0), "ms")
                        recognize(image, boxes[detected_index], faces[existing_index])
                        print("After recognition:", 1000*(time.time()-t0), "ms")
                        
                        
                    faces[existing_index].detected(boxes[detected_index],
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
        for index, box in enumerate(boxes):
            print("WARNING NEW FACE")
            p = Perso()
            p.detected(box, landmarks[index], frame_index)
            recognize(image, box, p)
            faces.append(p)
            
        
            
        # We loop through the existing faces to tag the faces that did not get detected
        index = 0
        while index < len(faces):
            face = faces[index]
            incrementIndex = True
            if face.last_frame_update < frame_index:
                face.nb_times_missed += 1
                
                if face.nb_times_missed >= nb_times_missed_threshold:
                    faces.pop(index)
                    incrementIndex = False
                    
            if incrementIndex: index += 1

        for face in faces:
            face.draw(image, frame_index)
        
        
        for i, face_landmarks in enumerate(results.multi_face_landmarks):    
            """
            cropped = image[new_cy_min:new_cy_max, new_cx_min:new_cx_max]
            small_frame = cv2.resize(cropped, (0, 0), fx=0.4, fy=0.4)
    
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]
            face_locations = face_recognition.face_locations(rgb_small_frame)
            list_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations) 
            
            if len(list_encodings) > 0:
                unknown_face_encoding = list_encodings[0]
                results = face_recognition.face_distance(detector.known_face_encodings, unknown_face_encoding)
                #print(results)
            #detector.analyze(cropped, frame_index)
            #detector.draw(image, frame_index, shift=(new_cx_min, new_cy_min))
            """
            
            # Mesh of the skull
            
            """
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_tesselation_style())
            # Outlines
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_contours_style())
            
            help(mp_face_mesh)
            # Eye balls
            
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_iris_connections_style())
            """
            
            # Draw contours of mouth by hand
            """
            contours = []
            for key in dict(mp_face_mesh.FACEMESH_LIPS):
                val = dict(mp_face_mesh.FACEMESH_LIPS)[key]
                landmark = face_landmarks.landmark[val]
                contours.append((int(landmark.x * frame_width), int(landmark.y * frame_height)))

                
            cv2.drawContours(image, [np.array(contours)], 0, (0,255,0), -1)
            """
            
            """
            mouth_pos = getMouthPosFromLandmark(face_landmarks.landmark, frame_width, frame_height)
            #cv2.ellipse(image, mouth_pos, (10,5), 10, 360, (0,0,255), -1)
            center = mouth_pos
            axes = (10,5)
            angle = 360
            cv2.ellipse(image, center, axes, angle, 0 , 360, (255,0,0), 2)
            """
            
            # Mouth
            """
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_LIPS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_contours_style())
            """
        
            #cv2.rectangle(image, (new_cx_min, new_cy_min), (new_cx_max, new_cy_max), (255, 255, 0), 2)
    
    # Flip the image horizontally for a selfie-view display.
    
    print("Compute frame:", 1000*(time.time()-t0), "ms")
    cv2.imshow('Main FacesDetector', image)
    print("Show frame:", 1000*(time.time()-t0), "ms")
    if cv2.waitKey(30) & 0xFF == 27:
      break
cap.release()





"""
# Make box bigger
        coef = 0.5
        w = cx_max - cx_min
        h = cy_max - cy_min
        new_cx_min = max(0, int(cx_min - coef * w))
        new_cy_min = max(0, int(cy_min - coef * h))
        new_cx_max = int(cx_max + coef * w)
        new_cy_max = int(cy_max + coef * h)
        
        x = int((cx_min + cx_max) / 2.)
        y = int((cy_min + cy_max) / 2.)
        boxes.append((x,y,w,h))
"""