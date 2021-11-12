from math import sqrt



def dist(x1, x2):
    return sqrt((x1[0]-x2[0])**2 + (x1[1]-x2[1])**2)
    

def getBoxFromLandmark(landmark, frame_width, frame_height):
    cx_min =  frame_width
    cy_min = frame_height
    cx_max = cy_max = 0
        
        
    for id, lm in enumerate(landmark):
        cx, cy = int(lm.x * frame_width), int(lm.y * frame_height)

        if cx < cx_min:
            cx_min = cx
        if cy < cy_min:
            cy_min = cy
        if cx > cx_max:
            cx_max = cx
        if cy > cy_max:
            cy_max = cy
                
    # From top-left/bottom-right ------> To Center/width-height
    w = cx_max - cx_min
    h = cy_max - cy_min
    x = int((cx_min + cx_max) / 2.)
    y = int((cy_min + cy_max) / 2.)
    return (x,y,w,h)