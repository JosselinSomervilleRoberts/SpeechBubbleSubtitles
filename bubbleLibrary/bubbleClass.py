from bubbleLibrary.utils_cv2 import rounded_rectangle, dist
import numpy as np
import cv2

class Bubble:
    
    def __init__(self):
        self.computed = False       # if the bubble was computed
        self.center = (0, 0)        # Center of bubble
        self.width = 0              # Width
        self.height = 0             # Height
        self.lines = []             # Lines of text to display
        self.font_scale = 0.5       # Font scale
        self.attach_mouth = (0, 0)  # Coordinates of the attach point (mouth)
        self.attach_bubble = (0, 0) # Coordinates of the attach of tail on the bubble
        self.perso = None           # Character
        self.frame_end = -1
        
    
    def initiate(self, center, width, height, lines, attach_mouth, frame_end = -1, perso = None):
        self.center = center
        self.width = width
        self.height = height 
        self.lines = lines
        self.attach_mouth = attach_mouth
        self.frame_end = frame_end
        self.perso = perso

    #-------
    #Getters
    #-------

    def getWidth(self):
        return self.width

    def getHeight(self):
        return self.height

    #-------
    #Setters
    #-------

    def setAttachMouth(self, new_attach_mouth):
        self.attach_mouth = new_attach_mouth

    #---------
    #Functions
    #---------

    def findAttachPosition(self, list_of_frames = None):
        #find best position for attach on bubble
        #put dialog in several lines if too big
        """Finds where to place the attach on the bubble"""
        # self.font_scale =
        #self.attach_bubble = 
        self.computed = True

    def drawBubble(self, frame, attach):
        """
        Draws a bubble
        input:
            frame: opencv image
            center: center of the bubble (tuple of ints)
            width, height: width and height of the bubble
            attach: position of the end of the bubble tail (tuple)
        """

        top_left     = (self.center[0] - int(self.width/2.), self.center[1] - int(self.height/2.))
        bottom_right = (self.center[0] + int(self.width/2.), self.center[1] + int(self.height/2.))
        #print(top_left, bottom_right)

        # Parameters
        radius           = 0.5
        outline_color    = (255, 0,   0  )
        fill_color       = (255, 255, 255)
        thickness        = 5
        line_type        = cv2.LINE_AA
        
        shapes = np.zeros_like(frame, np.uint8)
        #rounded_rectangle(img, top_left, bottom_right, radius=0.5, outline_color=(0,0,0), fill_color=(255,255,255), thickness=5, line_type=cv2.LINE_AA)
        rounded_rectangle(shapes,
                        top_left,
                        (bottom_right[1], bottom_right[0]),
                        radius = radius,
                        outline_color = outline_color,
                        fill_color = fill_color,
                        thickness = thickness,
                        line_type = line_type)
        
        x_tail_per = 0.8

        def percent_tail_to_absolute(x_per):
            return top_left[0] + radius*int(self.height/2.) + int(x_per * (self.width-int(radius*self.height/2.)))
        
        x_tail = percent_tail_to_absolute(x_tail_per)
        w_tail_half = int(0.8 * (1 + min(1.2, abs(x_tail-attach[0]) / (1+(abs(attach[1] - bottom_right[1]))))) * (0.1 * dist(attach, (x_tail, bottom_right[1]))))
        x_tail_1 =  int(max(x_tail - w_tail_half, top_left[0]     + radius*int(self.height/2.)))
        x_tail_2 =  int(min(x_tail + w_tail_half, bottom_right[0] - radius*int(self.height/2.)))
        list_points = [(x_tail_1, bottom_right[1]), (x_tail_2, bottom_right[1]), attach]
        #print(list_points)
        
        cv2.drawContours(shapes, [np.array(list_points)], 0, fill_color, -1)
        cv2.drawContours(shapes, [np.array(list_points)], 0, outline_color, thickness)
        #cv2.line(shapes, (100,200), (200,300), (255,255,255), 5, cv2.LINE_AA)
        alpha = 0.5
        mask = shapes.astype(bool)
        frame[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]

    def cutLinesIntoWords(self):
        """Returns the list of words in self.lines"""
        list_of_words = [""]
        lines_index = 0
        while lines_index < len(self.lines):
            #if a \n or \N is found (raw or normal), transform it into the \n word. We assume that whenever there is a \ in the text it is followed by n or N.
            if "\\" in r"%r" % self.lines[lines_index:lines_index+1]:
                list_of_words.append("\n")
                list_of_words.append("")
                lines_index +=2
            elif "\\n" in r"%r" % self.lines[lines_index:lines_index+1] or "\\N" in r"%r" % self.lines[lines_index:lines_index+1]:
                list_of_words.append("\n")
                lines_index += 1
            #else, just add the character to the current word unless it is a space
            else:
                if self.lines[lines_index] == ' ':
                    list_of_words.append("")
                else:
                    list_of_words[-1] += self.lines[lines_index]
                lines_index += 1
        return list_of_words

    def newLinePosition(word):
        """Finds the position (if it exists) of a new line. Should work for strings and raw strings"""
        for i in range(len(word)-1):
            if "\\" in r"%r" % word[i:i+1] or "\\n" in r"%r" % word[i:i+1] or "\\N" in r"%r" % word[i:i+1]:
                return i
        return -1


    def drawText(self, frame):
        """Draws the text inside the bubble"""
        text_per_line = [""]
        list_of_words = self.cutLinesIntoWords()
        
        size_space = cv2.getTextSize(" ", fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = self.font_scale, thickness = 2)[0][0]

        margin = 20
        empty_space_on_line = self.width - margin #size of empty space that can be filled with words
        for word in list_of_words:
            #if word is a \n, just change line
            if word == "\n":
                text_per_line.append("")
                empty_space_on_line = self.width - margin

            else:                
                size_word = cv2.getTextSize(word, fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = self.font_scale, thickness = 2)[0][0]
                #if there is not enough space on the current line, change line and reset the space left on the line
                if size_word > empty_space_on_line:
                    text_per_line.append("")
                    empty_space_on_line = self.width - margin

                text_per_line[-1] += word + " "
                empty_space_on_line -= size_word + size_space

        nb_lines = len(text_per_line)
        for index_line in range(nb_lines):
            size_text = cv2.getTextSize(text_per_line[index_line], fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = self.font_scale, thickness = 2)[0][0]
            cv2.putText(frame, text_per_line[index_line], org = (int(self.center[0]-0.5*size_text), int(self.center[1] - (nb_lines-1)*0.5*(self.height - margin) / nb_lines + index_line*(self.height - margin) / nb_lines)),
        			    fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = self.font_scale, color = (0, 0, 0), thickness = 2)

    ###trouver une size min acceptable pour la bulle en fonction du texte dans la bulle
    def draw(self, frame, attach_mouth):
        """Draw the bubble and the text inside it"""
        #Draw bubble
        self.attach_mouth = attach_mouth
        self.drawBubble(frame, self.attach_mouth)

        #Draw text
        self.drawText(frame)