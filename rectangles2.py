import numpy as np
import matplotlib.pyplot as plt
from math import sqrt


def rectangle_size(matrix, axisY=1, axisX=1):
    # for each point in the matrix returns the area of the rectange of maximum size centered at this point
    # the matrix given as an argument is binary : 
    # - 0 = space available for a rectangle
    # - 1 = space not available.
    matrix = matrix[::-axisY, ::-axisX]

    # Only accepts rectangle such as h/v >= RATIO_VERTICAL_MAX
    RATIO_VERTICAL_MAX = 1
    
    # Space going sideway and up-down-way
    h = np.zeros(matrix.shape, dtype=int)
    v = np.zeros(matrix.shape, dtype=int)
    
    for index in range(matrix.shape[0]):
        
        # FIRST: starting at diagonal element (index, index) and going to the right -> to (index, n-1)
        # THEN:  starting below diagonal element (index+1, index) and going down    -> to (n-1, index)
        for time in range(2):
            for ind2 in range(index+time, matrix.shape[1]):
                i, j = index, ind2 
                if time == 1: i, j = ind2, index
                
                if matrix[i,j] == 1:
                    pass # We keep 0 as a value
                else:
                    if i==0: # First line of the matrix
                        if j==0: # First cell of the matrix
                            h[0,0], v[0,0] = 1, 1
                        else:
                            h[0,j] = h[0, j-1] + 1
                            v[0,j] = 1
                    else:
                        if j==0: # First column (but not first cell)
                            h[i,0] = 1
                            v[i,0] = v[i-1, 0] + 1
                        else: # not in first line or column
                            a1 = 1 + h[i, j-1] # Single horizontal line
                            a2 = 1 + v[i-1, j] # Single vertical line
                            h3 = min(h[i-1, j], 1 + h[i, j-1]) # Combine rects from left and up
                            v3 = min(v[i, j-1], 1 + v[i-1, j]) # Combine rects from left and up
                            #v3 = min(v3, int(h3 * RATIO_VERTICAL_MAX))
                            a3 = h3 * v3 # Combine rects from left and up
                            if a1 >= a2 and a1 > a3:
                                h[i,j] = a1
                                v[i,j] = 1
                            elif a2 > a3:
                                h[i,j] = 1
                                v[i,j] = a2
                            else:
                                h[i,j] = h3
                                v[i,j] = v3
            
    matrix = matrix[::-axisY, ::-axisX]                    
    return h[::-axisY, ::-axisX], v[::-axisY, ::-axisX]


def get_matrix_obstacles(nx, ny, boxes=None):
    if boxes is None: boxes = []
    
    m = np.zeros((ny, nx), dtype=int)
    
    for box in boxes:
        (x,y,w,h) = box
        x = int(x*nx)
        y = int(y*ny)
        w = int(w*nx)
        h = int(h*ny)
        m[y:y+h, x:x+w] = 1
        
    return m


def get_matrix_distance(nx, ny, mouth_pos, head_box, padding=0.2):
     m = np.zeros((ny, nx))
      
     for i in range(ny):
         for j in range(nx):
             m[i,j] = sqrt( ((j/float(nx)) - mouth_pos[0])**2 + ((i/float(ny)) - mouth_pos[1])**2 )
             
     x0 = max(0, int(nx*(head_box[0] - padding * head_box[2])))
     y0 = max(0, int(ny*(head_box[1] - padding * head_box[3])))
     x1 = min(nx - 1, int(nx*(head_box[0] + (1.+padding) * head_box[2])))
     y1 = min(ny - 1, int(ny*(head_box[1] + (1.+padding) * head_box[3])))
     m[y0:y1, x0:x1] = 2.
     return m
 
    
def get_matrix_size(nx, ny, obstacles, mouth_pos):
    x_mouth = int(nx*mouth_pos[0])
    y_mouth = int(ny*mouth_pos[1])
    
    h = np.zeros((ny, nx), dtype=int)
    v = np.zeros((ny, nx), dtype=int)
    area = np.zeros((ny, nx), dtype=int)
    
    params = [(0,y_mouth, 0, x_mouth, -1, -1),
              (0,y_mouth, x_mouth,nx, +1, -1),
              (y_mouth,ny, 0,x_mouth, -1, +1),
              (y_mouth,ny, x_mouth,nx, +1, +1)]
    for param in params:
        (y0,y1,x0,x1,axisX,axisY) = param
        h_small, v_small = rectangle_size(obstacles[y0:y1, x0:x1], axisY, axisX)
        h[y0:y1, x0:x1] = h_small
        v[y0:y1, x0:x1] = v_small
        area[y0:y1, x0:x1] = h_small * v_small
        
    return h, v, area


def get_matrix_preffered_pos(nx, ny, mouth_pos, head_box, padding=0.5):
    m = np.zeros((ny, nx))
    
    for i in range(ny):
         for j in range(nx):
             fx = 1
             if (head_box[0] - padding*head_box[2]) * nx <= j <= nx * (head_box[0] + (1+padding)*head_box[2]):
                 fx = 0.5 * (1. + np.cos(2*np.pi * (j - nx * (head_box[0]-padding*head_box[2])) / (nx*(1+2*padding)*head_box[2])))
             if i <= ny*mouth_pos[1]:
                 m[i,j] = fx * np.sin(5.5*np.pi/6. * i/float(ny*mouth_pos[1]))
             else:
                 m[i,j] = fx * 0.5*(ny-1-i) / float(ny-1-mouth_pos[1])
             
    return m



def fill_center(matrix):
   h_centered = np.zeros(matrix.shape, dtype=int)
   v_centered = np.zeros(matrix.shape, dtype=int)

   for axisX in range(-1, 2, 2):
        for axisY in range(-1, 2, 2):
           h,v = rectangle_size(matrix, axisY, axisX)
           print(v)

           for i in range(h.shape[0]):
               for j in range(h.shape[1]):
                   width  = h[i,j]
                   height = v[i,j]
                   
                   index_i = [i - axisY*int((height-1)/2.)]
                   if height % 2 == 0:
                       inc = int(height/2.)
                       index_i = [i - axisY*(inc-1), i - axisY*inc]
                       height-=1
                   
                   index_j = [j - axisX*int((width-1)/2.)]
                   if width % 2 == 0:
                       inc = int(width/2.)
                       index_j = [j - axisX*(inc-1), j - axisX*inc]
                       width-=1
                   
                   for new_i in index_i:
                       for new_j in index_j:
                           old_a = h_centered[new_i, new_j] * v_centered[new_i, new_j]
                           new_a = width * height
                           if new_a > old_a:
                               h_centered[new_i, new_j] = width
                               v_centered[new_i, new_j] = height
                       
   print(h_centered * v_centered)
   for i in range(h.shape[0]):
       for j in range(h.shape[1]):
           if h_centered[i,j] == 0 and v_centered[i,j] == 0:
               h_centered[i,j] = - 20
               v_centered[i,j] = 20
   plt.matshow(h_centered * v_centered)
        
        




def rectangle_size_centered(matrix):
    # -1, -1 -> Going down right (point represents top left)
    # -1, +1 -> Going down left  (point represents top right)
    # +1, -1 -> Going up right   (point represents bottom left)
    # +1, +1 -> Going up left    (point represents bottom right)
    
    list_h, list_v = [], []
    for i in range(-1, 2, 2):
        for j in range(-1, 2, 2):
            h,v = rectangle_size(matrix, i, j)
            list_h.append(h)
            list_v.append(v)
            print(h*v)
            print("\n")
            
    h = np.minimum.reduce(list_h)
    v = np.minimum.reduce(list_v)     
    return h,v


if __name__ == '__main__':
    nx, ny = 50, 50
    mouth_pos = (0.5,0.5)
    head_box = (0.4, 0.4, 0.4, 0.2)
    boxes = [head_box]
    boxes.append((0.1,0.2,0.2,0.25))
    obstacles = get_matrix_obstacles(nx, ny, boxes)
    h, v, area = get_matrix_size(nx, ny, obstacles, mouth_pos)
    distance = get_matrix_distance(nx, ny, mouth_pos, head_box)
    optimal_pos = get_matrix_preffered_pos(nx, ny, mouth_pos, head_box)
    
    plt.title("Obstacles")
    plt.matshow(obstacles)
    #plt.subplot(1,2,1)
    plt.matshow(area)
    
    #plt.subplot(1,2,2)
    plt.matshow(distance)
    
    plt.matshow(optimal_pos)
    
    plt.matshow(area * (2-distance)**5 * optimal_pos)
    
    """
    m = np.zeros((20,20))
    m[5,5] = 1
    m[2,2] = 1
    #h,v = rectangle_size(m)
    fill_center(m)
    """
    """
    #m[2,2] = 1
    #m[3,9] = 1
    h, v = rectangle_size_centered(m)
    print('h')
    print(h)
    print('\nv')
    print(v)
    print("\narea")
    print(h*v)
    """