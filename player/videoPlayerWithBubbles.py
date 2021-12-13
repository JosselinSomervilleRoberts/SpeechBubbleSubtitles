from player.videoPlayer import VideoPlayer
from recognizer.face import Face
from recognizer.utils import dist
from recognizer.interpolable import Interpolable
from recognizer.recognizer import Recognizer
from recognizer.utils import getBoxFromLandmark
from bubbleLibrary.read_files import read_subtitles
from posFinder import find_optimal_pos, display_results

from math import sqrt
import numpy as np
import cv2
import mediapipe as mp

from bubbleLibrary import bubbleClass




class VideoPlayerWithBubbles(VideoPlayer):
    
    # similarity required to consider that two faces are the same
    SIMILIRATY_THRESHOLD = 0.5 # between 0 (completly different) and 1 (exact same place and size)

    # How often we run the recognition function
    REFRESH_RECOGNITION = 10 # frames

    # Used to clean up the cases in finish_process
    MAXIMUM_NUMBER_OF_FRAMES_TO_JOIN_KNOWN_FACES = 8
    MAXIMUM_NUMBER_OF_FRAMES_TO_JOIN_UNKNOWN_FACES = 4
    MAXIMUM_DISTANCE_TO_JOIN_UNKNOWN_FACES = 0.04
    FACE_NOT_APPEARED_MAX_NUMBER_FRAMES = 5
    FACE_MINIMUM_APPEARED_NUMBER_FRAMES = 10

    # How often we run finish_process
    REFRESH_FINISH_PROCESS = 5 # frames
    DELAY_FINISH_PROCESS = 20  # frames

    # RESOLUTION
    RESOLUTION_DIV = 5
    RESOLUTION_WIDTH_MIN = 426
    RESOLUTION_HEIGHT_MIN = 240
    

    def __init__(self, video_path, subtitle_path, face_path):
        VideoPlayer.__init__(self, video_path)
        print("FPS=", self.fps)

        # Global parameters transmission
        # TODO: Fix this
        # Interpolable.MAX_INTERPOLATION_INTERVAL = VideoPlayer.MAXIMUM_NUMBER_OF_FRAMES_TO_JOIN_KNOWN_FACES + 1
        
        # Video data
        self.subtitle_path = subtitle_path
        self.subtitles = []
        self.subtitles_process = read_subtitles(subtitle_path, self.fps)

        # Bubbles
        self.bubbles = []
        
        # Display option
        self.show_mesh = False
        self.show_outline = True
        self.show_faces = True
        
        # General frame infos
        self.frame_width = None
        self.frame_height = None
        self.frame_width_reduced = None
        self.frame_height_reduced = None
        self.frame_normalizer = None
        
        # Face mesh and landmarks
        self.faces = []
        self.face_data  = []
        self.recognizer = Recognizer(face_path)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                            max_num_faces=3,
                            refine_landmarks=True,
                            min_detection_confidence=0.2,
                            min_tracking_confidence=0.4)

    
    
    def process_frame(self, frame, frame_index):
        
        # We have never loaded the general frame infos
        if self.frame_width is None:
            self.frame_width  = frame.shape[1]
            self.frame_height = frame.shape[0]
            self.frame_width_reduced  = max(self.frame_width  / VideoPlayerWithBubbles.RESOLUTION_DIV, VideoPlayerWithBubbles.RESOLUTION_WIDTH_MIN )
            self.frame_height_reduced = max(self.frame_height / VideoPlayerWithBubbles.RESOLUTION_DIV, VideoPlayerWithBubbles.RESOLUTION_HEIGHT_MIN)

            self.frame_normalizer = sqrt(self.frame_width_reduced**2 + self.frame_height_reduced**2) # To have distance between 0 and 1

            Face.frame_width  = self.frame_width
            Face.frame_height = self.frame_height
            Face.frame_normalizer = self.frame_normalizer
        
        frame = cv2.resize(frame, (self.frame_width_reduced, self.frame_height_reduced))

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
            boxes = [getBoxFromLandmark(landmark, self.frame_width_reduced, self.frame_height_reduced) for landmark in landmarks] # Bounding boxes of heads
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
                    index_max = np.unravel_index(arg_max, sim_mat.shape)
                    existing_index = available_faces_indexes[index_max[0]]
                    detected_index = index_max[1]
                    
                    # If the similarity is enough, we assign the detection to the existing face
                    if sim_mat[index_max] > VideoPlayerWithBubbles.SIMILIRATY_THRESHOLD:
                        
                        # if the face is unidentified we try to recognize it every so often
                        if ((frame_index + index_max[0]) % VideoPlayerWithBubbles.REFRESH_RECOGNITION == 0) and self.faces[existing_index].name is None:
                            self.recognizer.recognize(frame, boxes[detected_index], self.faces[existing_index])
                            
                            
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
                #self.recognizer.recognize(frame, box, new_face)
                self.faces.append(new_face)
                
            
            """
            # We loop through the existing faces to tag the faces that did not get detected
            index = 0
            while index < len(self.faces):
                face = self.faces[index]
                keep, delete = face.checkState(frame_index)

                if not(keep):
                    self.faces.pop(index)
                else:
                    index +=1
            """

        # Every so often, we cleanup the faces array
        if (frame_index > VideoPlayerWithBubbles.DELAY_FINISH_PROCESS) and (frame_index % VideoPlayerWithBubbles.REFRESH_FINISH_PROCESS == 0):
            self.finish_process(frame_index - VideoPlayerWithBubbles.DELAY_FINISH_PROCESS)

        if self.subtitles_process[0]["end"] < frame_index:
            self.process_subtitle(self.subtitles_process.pop(0))



    def process_subtitle(self, subtitle):
        boxes = []
        start = subtitle["start"]
        end = subtitle["end"]
        name = "raymond"
        if not(subtitle["name"] is None) and len(subtitle["name"]) > 0:
            name = subtitle["name"]

        face_index = -1
        for (i, face) in enumerate(self.faces):
            boxes += face.get_trace(start, end, self.frame_width_reduced, self.frame_height_reduced)
            if not(face.name is None) and face.name.lower() == name: face_index = i

        mouth_pos = (0.5, 1.0) # We'll keep the lowest mouth pos found
        head_box = (0.4, 0.4, 0.2, 0.2)
        if face_index < 0: # We did not find who is speaking
            mouth_pos = (0.5, 0.5)
        else:
            face = self.faces[face_index]
            subtitle["face_id"] = face.id
            for frame_index in range(start, end):
                if face.isPresent(frame_index):
                    new_mouth_pos = face.getMouthPos(frame_index)
                    if new_mouth_pos[1] >=0 and new_mouth_pos[1] / self.frame_height_reduced < mouth_pos[1]:
                        mouth_pos = (new_mouth_pos[0] / self.frame_width_reduced, new_mouth_pos[1] / self.frame_height_reduced)
                        head_box = face.getBox(frame_index)
            

        w = 0.4
        h = 0.25
        #print(boxes)
        subtitle["attach"] = (len(self.faces) > 0)
        if len(self.subtitles) > 0 and self.subtitles[-1]["end"] > start:
            boxes.append((self.subtitles[-1]["pos"][0] - w/2., self.subtitles[-1]["pos"][1] - h/2., self.subtitles[-1]["pos"][0] + w/2., self.subtitles[-1]["pos"][1] + h/2.))
            pass#head_box.append
        #display_results(mouth_pos, head_box, boxes, w, h, int(self.frame_width / 4.), int(self.frame_height / 4.))
        pos = find_optimal_pos(mouth_pos, head_box, boxes, w, h, self.frame_width_reduced, self.frame_height_reduced)
        subtitle["pos"] = pos
        subtitle["w"] = w
        subtitle["h"] = h
        subtitle["perso"] = name
        self.subtitles.append(subtitle)


    def finish_process(self, frame_index):
        """
        This function postprocesses the array self.faces.
        It looks for faces no longer active at the time of frame_index (meaning they
        have not been detected for at least VideoPlayerWithBubbles.FACE_NOT_APPEARED_MAX_NUMBER_FRAMES)
        These faces are tested on several aspects :
        - if they are named, we check for faces with the same name to see if we can merge the 2,
        or unnamed face that are close to merge the 2
        - if they are unnamed, we look for any faces (named or not) that were close to merge the 2
        - if even after merging a face only appears for a very short period, then we delete it
        TODO: Check for scene cuts before merging
        """

        face_index = 0
        while face_index < len(self.faces):
            face = self.faces[face_index]

            # We will treat the faces no longer active
            if frame_index - face.last_appearance > VideoPlayerWithBubbles.FACE_NOT_APPEARED_MAX_NUMBER_FRAMES:
                # We see if we can merge it to an other face

                # If the face has been identified, we will try to find a face with the same name
                if not(face.name is None):

                    face2_index = face_index + 1
                    while face2_index < len(self.faces):
                        removed = False
                        face2 = self.faces[face2_index]

                        if face2.name == face.name: # Deux faces sont le meme personnage

                            if face2.first_appearance <= face.last_appearance: # On a 2 fois le meme personnage en meme temps, ca ne va pas
                                face2.name = None
                            else:
                                # TODO: verifier qu'il n'y a pas de cut
                                if face.last_appearance - face2.first_appearance <= VideoPlayerWithBubbles.MAXIMUM_NUMBER_OF_FRAMES_TO_JOIN_KNOWN_FACES: 
                                    # Il y a peu de temps qui s ecoule entre les deux, donc on va reunir les deux faces
                                    face.merge(face2)
                                    self.faces.pop(face2_index)
                                    removed = True

                        if not(removed): face2_index += 1


                # If the face has not been identified, then we look for a face that has been identified close to this one
                # to see if we can merge the two, in order to identify the current face
                # We also look for unidentified face like that we can have powerful merge in chains
                else: # Unidentified face
                    lastPos = face.getLastKnownPos()

                    face2_index = face_index + 1
                    while face2_index < len(self.faces):
                        removed = False
                        face2 = self.faces[face2_index]

                        # TODO: verifier qu'il n'y a pas de cut
                        if face2.first_appearance - face.last_appearance <= VideoPlayerWithBubbles.MAXIMUM_NUMBER_OF_FRAMES_TO_JOIN_UNKNOWN_FACES:
                            # Il y a peu de temps qui s ecoule entre les deux, on va donc vérifier si les faces sont proches
                            distance = dist(lastPos, face2.getFirstKnownPos()) / self.frame_normalizer
                            if distance <= VideoPlayerWithBubbles.MAXIMUM_DISTANCE_TO_JOIN_UNKNOWN_FACES:
                                face.merge(face2)
                                self.faces.pop(face2_index)
                                removed = True

                        if not(removed): face2_index += 1


                # Finally, now that we have checked if the face could be merged,
                # We check its duration and if it is too short, we consider this detection
                # as an error and remove the face
                if face.last_appearance - face.first_appearance < VideoPlayerWithBubbles.FACE_MINIMUM_APPEARED_NUMBER_FRAMES:
                    self.faces.pop(face_index)
                else:
                    # We did not remove the face, so we check the next one
                    face_index += 1
            else:
                # The face is still active, we go to the next one
                face_index += 1


    def prepare_display(self):
        """This function replaces the variable self.current_frame using the video for display (self.cap_display)
        and the information processed (self.pending[0])"""

        #afficher les bulles dans cette fonction (modifier self.current_frame)

        # Read the frame from video file
        VideoPlayer.prepare_display(self)

        # New bubble
        if len(self.subtitles) > 0 and self.current_frame_index >= self.subtitles[0]["start"]:
            sub = self.subtitles.pop(0)
            lines = sub["text"]
            frame_end = sub["end"]
            pos_mouth = (300,250)
            pos_bubble = (int(self.frame_width * sub["pos"][0]), int(self.frame_height * sub["pos"][1]))
            width = int(self.frame_width * sub["w"])
            height = int(self.frame_height * sub["h"])
            bubble = bubbleClass.Bubble()
            bubble.initiate(pos_bubble, width, height, lines, pos_mouth, sub["attach"], frame_end=frame_end, perso=sub["perso"])

            area = bubble.estimateOptimalBubbleArea(width)
            bubble.setWidthAndHeight(width, area / width)

            self.bubbles.append(bubble)
    

        face_index = 0
        perso_pos = {}
        while face_index < len(self.faces):
            face = self.faces[face_index]
            removed = False

            if face.isPresent(self.current_frame_index):
                if self.show_faces: face.draw(self.current_frame, self.current_frame_index)
                if not(face.name is None): perso_pos[face.name.lower()] = face.getMouthPos(self.current_frame_index)
            else:
                if self.current_frame_index > face.last_appearance: # La tete n'apparaitra plus
                    # On verifie juste qu'on n'en aura pas besoin pour le finish_process
                    if self.current_frame_index + len(self.pending) - face.last_appearance > VideoPlayerWithBubbles.DELAY_FINISH_PROCESS + VideoPlayerWithBubbles.REFRESH_FINISH_PROCESS:
                        self.faces.pop(face_index)
                        removed = True

            if not(removed): face_index += 1

        

        #Draw bubbles on the frame
        for bubble in self.bubbles:
            draw_tail = False
            if bubble.perso == "raymond" and not(bubble.perso in perso_pos):
                bubble.perso = "santiago"
            if bubble.perso == "kevin" and not(bubble.perso in perso_pos):
                bubble.perso = "diaz"
            if bubble.perso == "diaz" and not(bubble.perso in perso_pos):
                bubble.perso = "kevin"
            if bubble.perso is None or len(bubble.perso) == 0 or not(bubble.perso in perso_pos):
                # TODO: Change to search for perso speaking without names
                found = False
                for face in self.faces:
                    if not(found):
                        if face.isSpeaking(self.current_frame_index):
                            new_pos = face.getMouthPos(self.current_frame_index)
                if not(found):
                    new_pos = (0,0)
                    bubble.display_attach = False
                    #if len(self.faces) > 0:
                    #    new_pos = face.getMouthPos(self.current_frame_index)
            else:
                new_pos = perso_pos[bubble.perso]
                if new_pos[0] < 0: new_pos = (0,100)
            draw_tail = new_pos[0] > 0 #if mouth wasn't found (ie pos <= 0), we don't want to draw the tail of the bubble

                # print("Pos jake =", new_pos)
            bubble.setAttachMouth(new_pos)
            bubble.draw(self.current_frame, draw_tail = draw_tail)

        # Clean old bubbles
        ind = 0
        while ind < len(self.bubbles):
            if self.bubbles[ind].frame_end <= self.current_frame_index:
                self.bubbles.pop(ind)
            else:
                ind += 1
