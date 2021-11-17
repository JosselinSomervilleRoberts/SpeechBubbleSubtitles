from bubbleLibrary.utils_cv2 import rounded_rectangle, dist
import numpy as np
import cv2


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
    radius           = 0.5
    outline_color    = (255, 0,   0  )
    fill_color       = (255, 255, 255)
    thickness        = 5
    line_type        = cv2.LINE_AA
    
    shapes = np.zeros_like(frame, np.uint8)
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
        return top_left[0] + radius*int(height/2.) + int(x_per * (width-int(radius*height/2.)))
    
    x_tail = percent_tail_to_absolute(x_tail_per)
    w_tail_half = int(0.8 * (1 + min(1.2, abs(x_tail-attach[0]) / (1+(abs(attach[1] - bottom_right[1]))))) * (0.1 * dist(attach, (x_tail, bottom_right[1]))))
    x_tail_1 =  int(max(x_tail - w_tail_half, top_left[0]     + radius*int(height/2.)))
    x_tail_2 =  int(min(x_tail + w_tail_half, bottom_right[0] - radius*int(height/2.)))
    list_points = [(x_tail_1, bottom_right[1]), (x_tail_2, bottom_right[1]), attach]
    #print(list_points)
    
    cv2.drawContours(shapes, [np.array(list_points)], 0, fill_color, -1)
    cv2.drawContours(shapes, [np.array(list_points)], 0, outline_color, thickness)
    #cv2.line(shapes, (100,200), (200,300), (255,255,255), 5, cv2.LINE_AA)
    alpha = 0.5
    mask = shapes.astype(bool)
    frame[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]