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
        # self.lines = 
        self.perso = perso


    def findOptimalPosition(self, list_of_frames):
        """Find where to place the bubble"""
        # self.center   = 
        # self.width    =
        # self.height   =
        # self.fontsize =
        self.computed = True

    
    def draw(self, frame, attach1):
        """Draw the bubble"""
        self.attach1 = attach1
        modified_frame = frame
        return modified_frame


# old code

def draw_bubble_text(frame, center, width, height, attach):
    """
    input:
        frame: opencv image
        center: center of the bubble (tuple of ints)
        width, height: width and height of the bubble
        attach: position of the end of the bubble tail (tuple)
    """
    
    top_left     = (center[0] - int(width/2.), center[1] - int(height/2.))
    bottom_right = (center[0] + int(width/2.), center[1] + int(height/2.))
    print(top_left, bottom_right)
    
    # Parameters
    radius    = 0.5
    stroke    = (255, 0,   0  )
    fill      = (255, 255, 255)
    thickness = 5
    line_type = cv2.LINE_AA
    
    shapes = np.zeros_like(frame, np.uint8)
    rounded_rectangle(shapes,
                      top_left,
                      (bottom_right[1], bottom_right[0]),
                      radius = radius,
                      stroke = stroke,
                      fill   = fill,
                      thickness = thickness,
                      line_type = line_type)
    
    x_tail_per = 0.8

    def percent_tail_to_absolute(x_per):
        return top_left[0] + radius*int(height/2.) + int(x_per * (width-int(radius*height/2.)))
    
    x_tail = percent_tail_to_absolute(x_tail_per)
    w_tail_half = int(0.8 * (1 + min(1.2, abs(x_tail-attach[0]) / (1+(abs(attach[1] - bottom_right[1]))))) * (0.1 * dist(attach, (x_tail, bottom_right[1]))))
    x_tail_1 =  int(max(x_tail - w_tail_half, top_left[0]     + radius*int(height/2.)))
    x_tail_2 =  int(min(x_tail + w_tail_half, bottom_right[0] - radius*int(height/2.)))
    list_points = [(x_tail_1, bottom_right[1]), (x_tail_2, bottom_right[1]), attach]
    #print(list_points)
    
    cv2.drawContours(shapes, [np.array(list_points)], 0, fill, -1)
    cv2.drawContours(shapes, [np.array(list_points)], 0, stroke, thickness)
    #cv2.line(shapes, (100,200), (200,300), (255,255,255), 5, cv2.LINE_AA)
    alpha = 0.5
    mask = shapes.astype(bool)
    frame[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]