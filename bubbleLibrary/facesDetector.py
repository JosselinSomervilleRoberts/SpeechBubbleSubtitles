from bubbleLibrary.perso import Perso
import face_recognition
import cv2
import numpy as np


class FacesDetector:
    nb_times_missed_max = 3
    frame_width  = 0
    frame_height = 0
    
    def __init__(self):
        # Known faces
        self.known_face_encodings = []
        self.known_face_names     = []
        
        # Locations and landmarks of the faces
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        
        # Characters present in the scene
        self.persos = {}
        
        
    def addKnownFace(self, image_path, name):
        # Load image
        image    = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image, model="small")[0]
        
        # Add the encoding to the recognition
        self.addKnownFaceEncoding(encoding, name)
        return encoding

    def addKnownFaceEncoding(self, encoding, name):
        # Add the encoding to the recognition
        self.known_face_encodings.append(encoding)
        self.known_face_names.append(name)
        
    
    def preprocess(frame):
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = frame #cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
        
        return rgb_small_frame
    
    
    def update(self, frame, frame_index):
        pass
    
    
    def analyze(self, frame, frame_index):
        new_frame = FacesDetector.preprocess(frame)
        (FacesDetector.frame_height, FacesDetector.frame_width) = new_frame.shape[:2]
        
        # Find all the faces and face encodings in the current frame of video
        self.face_locations = face_recognition.face_locations(new_frame)
        self.face_encodings = face_recognition.face_encodings(new_frame, self.face_locations)
        print(len(self.face_encodings))
        
        # Find the names
        for face_encoding in self.face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]

            self.face_names.append(name)
            
    
        # We update the boxes of the characters
        # and create new characters that were not previously detected
        # and also keep track of the faces that were previously in the
        # scene but not detected this time
        for index, box in enumerate(self.face_locations):
            name = self.face_names[index]
            # We create the character
            if not(name in self.persos):
                p = Perso()
                p.name = name
                self.persos[name] = p
                
            # We notice the character that we detected it
            self.persos[name].detected(self.face_locations[index], frame_index)
            
        for name in self.persos:
            if not(name in self.face_names):
                self.persos[name].nb_times_missed += 1
                
        
        # We remove characters that are not present in the scene anymore
        """
        index = 0
        names = self.persos.keys()
        while index < len(names):
            perso = self.persos[name]#index]
            mouth_pos =  perso.get_mouth_pos(frame_index)
            
            # REMOVE CONDITION : predicted head position outside of scene
            if (mouth_pos[0] <= 0) or (mouth_pos[0] >= FacesDetector.frame_width-1) or (mouth_pos[1] <= 0) or (mouth_pos[1] >= FacesDetector.frame_height-1):
                del self.persos[name]
                #del names[index]
            
            # REMOVE CONDITION : head not detected since a few recognitions
            elif perso.nb_times_missed >= FacesDetector.nb_times_missed_max:
                del self.persos[name]
                #del names[index]
                
            # WE KEEP THE CHARACTER
            #else:
            index += 1
        """
                
                
    def draw(self, frame, frame_index, shift=(0,0)):
        
        for name in self.persos:
            perso = self.persos[name]
            # center / width-height coordinates
            x = perso.x.predict(frame_index)
            y = perso.y.predict(frame_index)
            w = perso.w.predict(frame_index)
            h = perso.h.predict(frame_index)
            print(perso.x.x)
            print(perso.x.computed)
            print(perso.x.x[0],y,w,h)
            
            # top-left / bottom-right coordinates
            top    = int(y - h/2.) + shift[1]
            left   = int(x - w/2.) + shift[0]
            bottom = int(y + h/2.) + shift[1] 
            right  = int(x + w/2.) + shift[0] 
            
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    
            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            print("name =", name)
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            
        return frame
            
            
 
    
 
    
"""
    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
"""

