import face_recognition
import numpy as np
import os


class Recognizer:

    DISTANCE_MAX = 0.55
    
    def __init__(self, faces_path = None):
        # Known faces
        self.known_face_encodings = []
        self.known_face_names     = []

        # Loading
        if not(faces_path is None): self.load(faces_path)
        
        
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
    


    def recognize(self, frame, box, face):
        # Enlarge the box to be sure to have the full face in the cropped frame
        enlarge_coef = 2.0
        new_cx_min = max(0, int(box[0] - 0.5 * enlarge_coef * box[2]))
        new_cy_min = max(0, int(box[1] - 0.5 * enlarge_coef * box[3]))
        new_cx_max = int(box[0] + 0.5 * enlarge_coef * box[2])
        new_cy_max = int(box[1] + 0.5 * enlarge_coef * box[3])

        # Crop the image to keep only the face
        cropped = frame[new_cy_min:new_cy_max, new_cx_min:new_cx_max]
        face_image = Recognizer.preprocess(cropped)

        # Encode the face
        face_locations = face_recognition.face_locations(face_image)
        list_encodings = face_recognition.face_encodings(face_image, face_locations, num_jitters=1, model="small")
        
        # if we found at least one face
        if len(list_encodings) > 0:

            index_min = -1
            min_distance = 1
            
            for encoding in list_encodings:
                distances = face_recognition.face_distance(self.known_face_encodings, encoding)
                
                index_min_for_this_encoding = np.argmin(distances)
                if distances[index_min_for_this_encoding] < min_distance:
                    min_distance = distances[index_min_for_this_encoding]
                    index_min = index_min_for_this_encoding

            # If we found an encoding close enough to an existing one, then we identify it     
            if min_distance < Recognizer.DISTANCE_MAX:
                face.name = self.known_face_names[index_min]



    def load(self, faces_path):
        print("========== Loading from " + faces_path + " =========")
        # list of characters available
        directories = [ name for name in os.listdir(faces_path) if os.path.isdir(os.path.join(faces_path, name)) ]

        for folder in directories:
            folder_path = os.path.join(faces_path, folder)
            encodings_path = os.path.join(folder_path, "encodings")

            # Create folder to save encoding
            if not(os.path.exists(encodings_path)): os.makedirs(encodings_path)

            image_files = [x for x in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, x))]
            encoding_files = [x.split(".")[0] for x in os.listdir(encodings_path) if os.path.isfile(os.path.join(encodings_path, x))]
            encoded_files_not_in_images = [x for x in encoding_files if not(x in [a.split(".")[0] for a in os.listdir(folder_path)])]
            total = len(image_files) + len(encoded_files_not_in_images)

            for index, file in enumerate(image_files):
                print("Loading " + folder + " [" + "="*(index+1) + " "*(total-1-index) + "] " + "0"*(len(str(total)) - len(str(1+index))) + str(1+index) + " / " + str(total), end="\r")
                file_without_extension = file.split(".")[0]
                if file_without_extension in encoding_files:
                    encoding = np.loadtxt(os.path.join(encodings_path, file_without_extension + ".txt"))
                    self.addKnownFaceEncoding(encoding, folder)
                else:
                    encoding = self.addKnownFace(os.path.join(folder_path, file), folder)
                    np.savetxt(os.path.join(encodings_path, file_without_extension + ".txt"), encoding)

            for ind, file in enumerate(encoded_files_not_in_images):
                index = ind + len(image_files)
                print("Loading " + folder + " [" + "="*(index+1) + " "*(total-1-index) + "] " + "0"*(len(str(total)) - len(str(1+index))) + str(1+index) + " / " + str(total), end="\r")
                encoding = np.loadtxt(os.path.join(encodings_path, file))
                self.addKnownFaceEncoding(encoding, folder)

            print("")
