from bubbleLibrary.utils_cv2 import rounded_rectangle, dist
import numpy as np
import cv2

class Bubble:
    
    def __init__(self):
        self.computed = False               # if the bubble was computed
        self.center = (0, 0)                # Center of bubble
        self.width = 0                      # Width
        self.height = 0                     # Height
        self.lines = []                     # Lines of text to display
        self.font_scale = 0.5               # Font scale
        self.attach_mouth = (0, 0)          # Coordinates of the attach point (mouth)
        self.attach_bubble = [(0,0), (0,0)] # Coordinates of the attach of the tail on the bubble
        self.perso = None                   # Character
        self.frame_end = -1                 # ???
        
    def initiateAttachBubble(self):
        """compute the attach of the tail on the bubble. The two points should belong to :
            - an edge of the bubble rectangle
            - the lines passing through the attach on the mouth and the center of the rectangle +/- a fraction of its height (called up and down in the code)
                -> we compute their equation as y=a_up*x+b_up and y=a_down*x+b_down
        
        """

        center_up = (self.center[0], self.center[1] - self.height/5.)
        center_down = (self.center[0], self.center[1] + self.height/5.)

        delta_x = self.center[0] - self.attach_mouth[0]
        if delta_x == 0:
            delta_x = 1e-5
        a_up = (center_up[1] - self.attach_mouth[1]) / delta_x
        b_up = center_up[1] - a_up*center_up[0]
        a_down = (center_down[1] - self.attach_mouth[1]) / delta_x
        b_down = center_down[1] - a_down*center_down[0]
        #compute the x coordinate on an edge of the rectangle depending on the mouth's position
        if self.attach_mouth[0] < self.center[0] - self.width/2.: # left edge
            #print(" left")
            x_up = self.center[0] - self.width/2.
            y_up = a_up*x_up + b_up
            x_down = self.center[0] - self.width/2.
            y_down = a_down*x_down + b_down 
        elif self.attach_mouth[0] > self.center[0] + self.width/2.: # right edge
            #print(" right")      
            x_up = self.center[0] + self.width/2.
            y_up = a_up*x_up + b_up
            x_down = self.center[0] + self.width/2.
            y_down = a_down*x_down + b_down 
        elif self.attach_mouth[1] > self.center[1]: # bottom edge
            #print(" bottom")
            y_up = self.center[1] + self.height/2.
            x_up = (y_up - b_up) / a_up
            y_down = self.center[1] + self.height/2.
            x_down = (y_down - b_down) / a_down
        else: #upper edge
            #print(" up")
            y_up = self.center[1] - self.height/2.
            x_up = (y_up - b_up) / a_up
            y_down = self.center[1] - self.height/2.
            x_down = (y_down - b_down) / a_down

        #y_up and y_down may be outside of the rectangle's range, so crop them if that's the case
        y_up = max(y_up, self.center[1] - self.height/5.)
        y_up = min(y_up, self.center[1] + self.height/5.)
        y_down = max(y_down, self.center[1] - self.height/5.)
        y_down = min(y_down, self.center[1] + self.height/5.)
        #self.attach_bubble[0] = (int(self.center[0]), int(self.center[1] - self.height/5.))
        #self.attach_bubble[1] = (int(self.center[0]), int(self.center[1] + self.height/5.))
        self.attach_bubble[0] = (int(x_up), int(y_up))
        self.attach_bubble[1] = (int(x_down), int(y_down))

    def initiate(self, center, width, height, lines, attach_mouth, frame_end = -1, perso = None):
        self.center = center
        self.width = width
        self.height = height 
        self.lines = lines
        self.attach_mouth = attach_mouth
        self.frame_end = frame_end
        self.perso = perso
        self.initiateAttachBubble()

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

    def setWidthAndHeight(self, new_width, new_height):
        self.width = new_width
        self.height = new_height

    #---------
    #Functions
    #---------
            #-----------------------------------
            #Determine the bubble's optimal area
            #-----------------------------------
    
    def estimateOptimalBubbleArea(self, bubble_width = 200):
        #find a good area for the bubble depending on the text
        #we take the approximate area that would be necessary to fit the text into a bubble of width bubble_width and add a 25% margin
        text_size = cv2.getTextSize(self.lines, fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = self.font_scale, thickness = 2)[0]
        nb_lines = text_size[0] // bubble_width + 1
        bubble_height = 2 * text_size[1] * nb_lines

        return 1.25 * bubble_width * bubble_height      

            #---------------
            #Draw the bubble
            #---------------

    def drawBubble(self, frame, draw_tail = True):
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

        # Parameters for the bubble
        radius           = 0.5
        outline_color    = (255, 0,   0  )
        fill_color       = (255, 255, 255)
        thickness        = 2
        line_type        = cv2.LINE_AA
        
        shapes = np.zeros_like(frame, np.uint8)

        #Draw the tail of the bubble
        if draw_tail:
            #it is the intersection of the triangle (mouth-attach_bubble[0]-attach_bubble[1]) and one of the sides of the rectangle
            self.initiateAttachBubble()
            attach_points = np.array([self.attach_bubble[0], self.attach_bubble[1], self.attach_mouth])

            cv2.drawContours(shapes, [attach_points], 0, fill_color, -1)
            cv2.drawContours(shapes, [attach_points], 0, outline_color, thickness)

        #Draw the rectangle for the bubble
        rounded_rectangle(shapes,
        top_left,
        (bottom_right[1], bottom_right[0]),
        radius = radius,
        outline_color = outline_color,
        fill_color = fill_color,
        thickness = thickness,
        line_type = line_type)

        #Add the bubble to the frame
        alpha = 0.5
        mask = shapes.astype(bool)
        frame[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]

            #-------------
            #Draw the text
            #-------------

    def cutLinesIntoWords(self):
        """Returns the list of words in self.lines. eg if self.lines = "Cool cool cool", returns ["Cool", "Cool", "Cool"]"""
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
        text_per_line = [""] #text we are going to display on each line
        list_of_words = self.cutLinesIntoWords()
        
        size_space = cv2.getTextSize(" ", fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = self.font_scale, thickness = 2)[0][0]

        margin = 20 #margin for text to leave some space at the boundary of the bubble
        empty_space_on_line = self.width - margin #size of empty space at the end of the line that can be filled with words
        for word in list_of_words:
            #if word is a \n, just change line
            if word == "\n":
                text_per_line.append("")
                empty_space_on_line = self.width - margin

            else:            
                height_word = cv2.getTextSize(word, fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = self.font_scale, thickness = 2)[0][1]
                size_word = cv2.getTextSize(word, fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = self.font_scale, thickness = 2)[0][0]
                #if there is not enough space on the current line, change line and reset the space left on the line
                if size_word > empty_space_on_line:
                    text_per_line.append("")
                    empty_space_on_line = self.width - margin
                #add current word to the right line
                text_per_line[-1] += word + " "
                empty_space_on_line -= size_word + size_space

        #display the right text on each line
        nb_lines = len(text_per_line)
        for index_line in range(nb_lines):
            size_text = cv2.getTextSize(text_per_line[index_line], fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = self.font_scale, thickness = 2)[0][0]
            y_org = int(self.center[1] - (nb_lines-1)*0.5*(self.height - margin) / nb_lines + index_line*(self.height - margin) / nb_lines)
            #y_org = int(self.center[0] + 2 * 12 * index_line)
            cv2.putText(frame, text_per_line[index_line], org = (int(self.center[0]-0.5*size_text), y_org),
        			    fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = self.font_scale, color = (0, 0, 0), thickness = 2)


            #---------------
            #Draw everything
            #---------------

    ###trouver une size min acceptable pour la bulle en fonction du texte dans la bulle
    def draw(self, frame, draw_tail = True):
        """Draw the bubble and the text inside it"""
        #Draw bubble
        self.drawBubble(frame, draw_tail)

        #Draw text
        self.drawText(frame)