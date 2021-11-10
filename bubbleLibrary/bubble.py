# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 11:40:02 2021

@author: josse
"""

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
        