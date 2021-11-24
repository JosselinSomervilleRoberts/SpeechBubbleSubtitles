import numpy as np

def horizontal_space(matrix):
    h1 = np.zeros(matrix.shape, dtype=int)
    h2 = np.zeros(matrix.shape, dtype=int)
    
    # For each line
    for i in range(matrix.shape[0]):
        h1[i,0] = 1-matrix[i,0]
        h2[i,-1] = 1-matrix[i,-1]
        for j in range(1,matrix.shape[1]):
            if matrix[i,j] == 1:
                h1[i,j] = 0
            else:
                h1[i,j] = 1 + h1[i,j-1]
                
            if matrix[i,-1-j] == 1:
                h2[i,-1-j] = 0
            else:
                h2[i,-1-j] = 1 + h2[i,-j]
    
    h = np.minimum(h1, h2)
    #h = np.maximum(np.zeros(matrix.shape), 2*h_min - 1)
    return h
    
    
def rectangle_size(matrix):
    # for each point in the matrix returns the area of the rectange of maximum size centered at this point
    # the matrix given as an argument is binary : 
    # - 0 = space available for a rectangle
    # - 1 = space not available.

    # Only accepts rectangle such as h/v >= RATIO_VERTICAL_MAX
    RATIO_VERTICAL_MAX = 1
    
    # Space going sideway and up-down-way
    h = horizontal_space(matrix)
    v = horizontal_space(matrix.T).T

    # Space available going upward
    h_av_up = np.zeros(matrix.shape, dtype=int)
    v_av_up = np.zeros(matrix.shape, dtype=int)

    for j in range(matrix.shape[1]):
        h_av_up[0,j] = max(0, 2*h[0,j]-1)
        v_av_up[0,j] = int(h[0,j] > 0)

        curr_h = h_av_up[0,j]
        curr_v = 1
        for i in range(1,matrix.shape[0]):
            if h[i,j] == 0:
                curr_v = 0
                curr_h = 0
            else:
                if 2*h[i,j]-1 > min(curr_h, 2*h[i,j]-1) * (curr_v+1): # Si c'est mieux de recommencer un rectangle
                    curr_h = 2*h[i,j]-1
                    curr_v = 1
                elif min(curr_h, 2*h[i,j]-1) / (2*(curr_v+1)-1) >= RATIO_VERTICAL_MAX:
                    curr_h = min(curr_h, 2*h[i,j]-1)
                    curr_v += 1
                else:
                    curr_h = min(curr_h, 2*h[i,j]-1)
                    # curr_v does not change
                
                h_av_up[i,j] = curr_h
                v_av_up[i,j] = curr_v


    # Space available going downward
    h_av_down = np.zeros(matrix.shape, dtype=int)
    v_av_down = np.zeros(matrix.shape, dtype=int)

    for j in range(matrix.shape[1]):
        h_av_down[-1,j] = max(0, 2*h[-1,j]-1)
        v_av_down[-1,j] = int(h[-1,j] > 0)
        
        curr_h = h_av_down[-1,j]
        curr_v = 1
        liste_h = [h_av_down[-1,j]]
        for ind in range(1,matrix.shape[0]):
            i = matrix.shape[0] - 1 - ind
            if h[i,j] == 0:
                curr_v = 0
                curr_h = 0
                liste_h = []
            else:
                if 2*h[i,j]-1 > min(curr_h, 2*h[i,j]-1) * (curr_v+1): # Si c'est mieux de recommencer un rectangle
                    curr_h = 2*h[i,j]-1
                    curr_v = 1
                    liste_h = [curr_h]
                elif min(curr_h, 2*h[i,j]-1) / (2*(curr_v+1)-1) >= RATIO_VERTICAL_MAX:
                    curr_h = min(curr_h, 2*h[i,j]-1)
                    curr_v += 1
                    liste_h.append(curr_h)
                else:
                    liste_h.pop(0)
                    liste_h.append(curr_h)
                    curr_h = min(curr_h, 2*h[i,j]-1)
                    # curr_v does not change
                
                h_av_down[i,j] = curr_h
                v_av_down[i,j] = curr_v

    # Space available centered
    h_av = np.zeros(matrix.shape, dtype=int)
    v_av = np.zeros(matrix.shape, dtype=int)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            h = min(h_av_down[i,j], h_av_up[i,j])
            v = 2*min(v_av_down[i,j], v_av_up[i,j]) - 1
            v = min(v, int(h/RATIO_VERTICAL_MAX))
            if h > 0 and v > 0:
                h_av[i,j] = h
                v_av[i,j] = v

    print(h_av_down)
    print("")
    print(h_av_up)
    print("")
    print(h_av)
    print("")
    print(v_av)
    print("")
    print(h_av*v_av)

    """
    print("h")
    print(h)
    print("\nv")
    print(v)    
    #space = 
    spv = np.zeros(matrix.shape, dtype=np.int)
    sph = np.zeros(matrix.shape, dtype=np.int)
              
    rect = np.zeros(matrix.shape, dtype=np.int)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if h[i,j] == 0:
                rect[i,j] = -1
            elif h[i,j] == 1:
                rect[i,j] = v[i,j]
            elif v[i,j] == 1:
                rect[i,j] = h[i,j]
            else: # h[i,j] > 1 && v[i,j] > 1
                v_cur = 1
                h_cur = 2*h[i,j] - 1
                continuer = True
                k = 1
                print("v[i,j] =", v[i,j], "   (i,j) = (" +str(i)  + "," + str(j) + ")")
                while continuer and k < v[i,j]:
                    print("k=", k)
                    h_min = max(min(h_cur, 2*min(h[i+k,j], h[i-k,j]) - 1), 0)
                    if (v_cur + 2) * h_min >= v_cur * h_cur and (v_cur+2) <= h_min:
                        v_cur += 2
                        h_cur = h_min
                    k += 1
                rect[i,j] = v_cur * h_cur
                spv[i,j] = v_cur
                sph[i,j] = h_cur
                
    print("\nrect")
    print(rect)
    print("\nsph")
    print(sph)
    print("\nspv")
    print(spv)
    """


if __name__ == '__main__':
    m = np.zeros((11,11))
    m[5,5] = 1
    #m[2,2] = 1
    #m[3,9] = 1
    rectangle_size(m)