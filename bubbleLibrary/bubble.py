from bubbleLibrary.utils_cv2 import rounded_rectangle, dist
import numpy as np
import cv2

class Bubble:
    
    def __init__(self):
        self.computed = False # if the bubble was computed
        self.center = (0, 0)  # Center of bubble
        self.width = 0        # Width
        self.height = 0       # Height
        self.lines = []       # Lines of text to display
        self.fontsize = 12    # Font size
        self.attach1 = (0, 0) # Coordinates of attach point (mouth)
        self.attach2 = (0, 0) # Coordinated of attach of tail on the bubble
        self.perso = None     # Character
        
    
    def initiate(self, text, perso = None):
        self.lines = text
        self.perso = perso

    #-------
    #Getters
    #-------

    def getWidth(self):
        return self.width

    def getHeight(self):
        return self.height

    #---------
    #Functions
    #---------

    def findOptimalPosition(self, list_of_frames = None):
        #1. find best width
        #2. put dialog in two lines if two big
        #3. limit width and height so that stays in frame
        """Finds where to place the bubble"""
        self.center    = (400, 100)
        #max_width      = 25 * self.fontsize
        self.width     = 2 * (len(self.lines)) * self.fontsize
        #self.width = 500
        self.height    = 100
        # self.fontsize =
        self.computed = True

    def draw_bubble(self, frame, attach):
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
        print(top_left, bottom_right)

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


    def draw(self, frame, attach1):
        """Draw the bubble and the text inside it"""
        #Draw bubble
        self.attach1 = attach1
        self.draw_bubble(frame, self.attach1)

        #Draw text
        cv2.putText(frame, self.lines, (int(self.center[0]-0.4*self.width), int(self.center[1] + 0.25 * self.height)),
        			cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        modified_frame = frame
        return modified_frame