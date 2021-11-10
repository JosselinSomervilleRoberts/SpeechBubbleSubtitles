# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 19:25:11 2021

@author: josse
"""

import cv2
import numpy as np


def dist(x1, x2):
    return ((x1[0]-x2[0])**2 + (x1[1]-x2[1])**2)**0.5

def rounded_rectangle(src, top_left, bottom_right, radius=0.5, stroke=(0,0,0), fill=(255,255,255), thickness=5, line_type=cv2.LINE_AA):

    #  corners:
    #  p1 - p2
    #  |     |
    #  p4 - p3
    p1 = top_left
    p2 = (bottom_right[1], top_left[1])
    p3 = (bottom_right[1], bottom_right[0])
    p4 = (top_left[0], bottom_right[0])


    # Corner radius
    height = abs(bottom_right[0] - top_left[1])
    if radius > 1:
        radius = 1
    corner_radius = int(radius * (height/2))
    
    
    # ========= FILL ======== #
    # draw rectangles
    top_left_main_rect = (int(p1[0] + corner_radius), int(p1[1]))
    bottom_right_main_rect = (int(p3[0] - corner_radius), int(p3[1]))
    top_left_rect_left = (p1[0], p1[1] + corner_radius)
    bottom_right_rect_left = (p4[0] + corner_radius, p4[1] - corner_radius)
    top_left_rect_right = (p2[0] - corner_radius, p2[1] + corner_radius)
    bottom_right_rect_right = (p3[0], p3[1] - corner_radius)
    all_rects = [
    [top_left_main_rect, bottom_right_main_rect], 
    [top_left_rect_left, bottom_right_rect_left], 
    [top_left_rect_right, bottom_right_rect_right]]
    [cv2.rectangle(src, rect[0], rect[1], fill, -1) for rect in all_rects]
    
    # draw arcs
    cv2.ellipse(src, (p1[0] + corner_radius, p1[1] + corner_radius), (corner_radius, corner_radius), 180.0, 0, 90, fill , -1, line_type)
    cv2.ellipse(src, (p2[0] - corner_radius, p2[1] + corner_radius), (corner_radius, corner_radius), 270.0, 0, 90, fill , -1, line_type)
    cv2.ellipse(src, (p3[0] - corner_radius, p3[1] - corner_radius), (corner_radius, corner_radius), 0.0, 0, 90,   fill , -1, line_type)
    cv2.ellipse(src, (p4[0] + corner_radius, p4[1] - corner_radius), (corner_radius, corner_radius), 90.0, 0, 90,  fill , -1, line_type)
    # ======================= #
    
    
    
    # ========= STROKE ======== #
    # draw arcs
    cv2.ellipse(src, (p1[0] + corner_radius, p1[1] + corner_radius), (corner_radius, corner_radius), 180.0, 0, 90, stroke , thickness, line_type)
    cv2.ellipse(src, (p2[0] - corner_radius, p2[1] + corner_radius), (corner_radius, corner_radius), 270.0, 0, 90, stroke , thickness, line_type)
    cv2.ellipse(src, (p3[0] - corner_radius, p3[1] - corner_radius), (corner_radius, corner_radius), 0.0, 0, 90,   stroke , thickness, line_type)
    cv2.ellipse(src, (p4[0] + corner_radius, p4[1] - corner_radius), (corner_radius, corner_radius), 90.0, 0, 90,  stroke , thickness, line_type)

    # draw straight lines
    cv2.line(src, (p1[0] + corner_radius, p1[1]), (p2[0] - corner_radius, p2[1]), stroke, thickness, line_type)
    cv2.line(src, (p2[0], p2[1] + corner_radius), (p3[0], p3[1] - corner_radius), stroke, thickness, line_type)
    cv2.line(src, (p3[0] - corner_radius, p4[1]), (p4[0] + corner_radius, p3[1]), stroke, thickness, line_type)
    cv2.line(src, (p4[0], p4[1] - corner_radius), (p1[0], p1[1] + corner_radius), stroke, thickness, line_type)
    # ========================= #
    
    return src