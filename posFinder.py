import numpy as np
from math import sqrt
from integratedImage import integrateMatrix


def score_distance(x):
    a = 10
    b = 0.5
    s = 0.2
    g = x + s
    f = (1/(b*g)**2 - a/(b*g)) * 0.1 *(1-x) / a
    return min(1, - 4.82*f)


def penalization_box(box):
    # TODO: change so that it depends of the class and prediction score
    return 1


def preference_score(x, y, mouth_pos, head_box, padding=0.5, border = 0.3):
    fx = 1
    if (head_box[0] - padding*head_box[2]) <= x <= (head_box[0] + (1+padding)*head_box[2]):
        fx = 0.5 * (1. + np.cos(2*np.pi * (x - (head_box[0]-padding*head_box[2])) / ((1+2*padding)*head_box[2])))
    fy = 1
    if y <= mouth_pos[1]:
        fy = np.sin((np.pi - np.arcsin(border)) * y/float(mouth_pos[1]))
    else:
        fy = border * (1-y) / float(1-mouth_pos[1])
    return fx * fy


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


def compute_score(m, i, j, w, h):
    (ny,nx) = m.shape
    i_up = i - h
    j_up = j - w
    if i_up < 0 or j_up < 0:
        return 0 # It does not fit
    s = m[i,j] - m[i_up, j] - m[i, j_up] + m[i_up, j_up]
    return s


def build_preference(mouth_pos, head_box, nx = 100, ny = 50):
    s = np.zeros((ny, nx))
    
    for i in range(ny):
        for j in range(nx):
            x = j/float(nx-1)
            y = i/float(ny-1)
            s[i,j] = preference_score(x, y, mouth_pos, head_box)
    return s


def build_distance(mouth_pos, nx = 100, ny = 50):
    s = np.zeros((ny, nx))
    
    for i in range(ny):
        for j in range(nx):
            x = j/float(nx-1)
            y = i/float(ny-1)
            dist = sqrt((x-mouth_pos[0])**2 + (y-mouth_pos[1])**2 )
            s[i,j] = score_distance(dist)
    return s


def build_obstacles(boxes, nx = 100, ny = 50):
    return get_matrix_obstacles(nx, ny, boxes)



def buildScore(boxes, mouth_pos, head_box, nx = 100, ny = 50, coeff_obst = 1.5):
    s = np.zeros((ny, nx))
    obstacles = get_matrix_obstacles(nx, ny, boxes)
    
    for i in range(ny):
        for j in range(nx):
            x = j/float(nx-1)
            y = i/float(ny-1)
            dist = sqrt((x-mouth_pos[0])**2 + (y-mouth_pos[1])**2 )
            s[i, j] = preference_score(x, y, mouth_pos, head_box) * score_distance(dist) - coeff_obst*obstacles[i,j]
    return s


def final_score(int_s, w, h):
    (ny, nx) = int_s.shape
    s = np.zeros((ny, nx))

    for i in range(ny):
        for j in range(nx):
            s[i,j] = compute_score(int_s, i, j, w, h)

    mini = np.min(s)
    s[np.isclose(s, 0)] = mini
    return s


    
def display_results():
    mouth_pos = (0.3, 0.5)
    nx = 100
    ny = 100
    head_box = (0.2, 0.4, 0.2, 0.2)
    boxes = [head_box] + [(0.1, 0.1, 0.1, 0.1), (0.5, 0.7, 0.3, 0.25), (0.8, 0.1, 0.1, 0.2), (0.5, 0.3, 0.2, 0.1), (0.02, 0.3, 0.15, 0.4)]
    width = 0.15
    height = 0.2
    
    
    matrices = [build_obstacles(boxes, nx, ny),
                build_distance(mouth_pos, nx, ny),
                build_preference(mouth_pos, head_box, nx, ny)]
                #buildScore(boxes, mouth_pos, head_box, nx, ny)]
    
    
    fig, axes = plt.subplots(2,3,figsize=(24,16),sharey=False,sharex=False,
                        gridspec_kw = {'height_ratios': [mat.shape[0] for mat in matrices[:2]]})
    axes[0][0].set_title("Obstacles $S_O$")
    axes[0][1].set_title("Distance to mouth $S_D$")
    axes[0][2].set_title("Preferred position $S_P$")
    axes[1][0].set_title("Combined score $S = S_P * S_D - \mu S_O$ with $\mu = 0.1$")
    axes[1][1].set_title("Cumulated score for $w=" + str(width) + "$ and $h=" + str(height) + "$")
    axes[1][2].set_title("Summary of input and output infos")
    
    
    for (i,a) in enumerate(axes[0]):
        img = a.matshow(matrices[i], vmin=0, vmax=1, cmap=plt.get_cmap("Greys"))
        plt.colorbar(img, ax=a)
        
    s = buildScore(boxes, mouth_pos, head_box, nx, ny)
    int_s = integrateMatrix(s)
    final_s = final_score(int_s, int(width * nx), int(height * ny))
    optimal_pos = np.unravel_index(final_s.argmax(), final_s.shape)
    
    img = axes[1][0].matshow(s, vmin=0, vmax=1, cmap=plt.get_cmap("Greys"))
    plt.colorbar(img, ax=axes[1][0])
    img = axes[1][1].matshow(final_s, cmap=plt.get_cmap("Greys"))
    plt.colorbar(img, ax=axes[1][1])
    
    img = axes[1][2].matshow(final_s, cmap=plt.get_cmap("Greys"))
    plt.colorbar(img, ax=axes[1][2])
    ax = axes[1][2]
    for box in boxes:
        #add rectangle to plot
        ax.add_patch(Rectangle((nx*box[0], ny*box[1]), nx*(box[2]), ny*(box[3]),
                edgecolor = 'red',
                fill=False,
                lw=1))

    box = head_box
    ax.add_patch(Rectangle((nx*box[0], ny*box[1]), nx*(box[2]), ny*(box[3]),
                edgecolor = 'blue',
                fill=False,
                lw=1))

    ax.add_patch(Circle((int(mouth_pos[0]*nx), int(mouth_pos[1]*ny)), nx/40.,
                edgecolor = 'purple',
                fill=False,
                lw=1))

    ax.add_patch(Circle((optimal_pos[1], optimal_pos[0]), nx/60.,
                edgecolor = 'green',
                fill=False,
                lw=1))

    ax.add_patch(Rectangle((optimal_pos[1] - 1*int(width * nx), optimal_pos[0] - 1*int(height * ny)), int(width * nx), int(height * ny),
                edgecolor = 'green',
                fill=False,
                lw=1))
    #axes[0].matshow(matrices[0], vmin=0, vmax=1, cmap=plt.get_cmap("Greys"))
    
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle
    
    display_results()
"""
    mouth_pos = (0.3, 0.5)
    nx = 100
    ny = 50
    head_box = (0.2, 0.4, 0.2, 0.2)
    boxes = [head_box] + [(0.1, 0.1, 0.1, 0.1), (0.5, 0.7, 0.3, 0.25), (0.8, 0.1, 0.1, 0.2)]
    width = 0.15
    height = 0.2

    s = buildScore(boxes, mouth_pos, head_box, nx, ny)
    int_s = integrateMatrix(s)
    final_s = final_score(int_s, int(width * nx), int(height * ny))
    optimal_pos = np.unravel_index(final_s.argmax(), final_s.shape)

    #define Matplotlib figure and axis
    fig, ax = plt.subplots()

    #create simple line plot
    ax.matshow(s)


    for box in boxes:
        #add rectangle to plot
        ax.add_patch(Rectangle((nx*box[0], ny*box[1]), nx*(box[2]), ny*(box[3]),
                edgecolor = 'red',
                fill=False,
                lw=1))

    box = head_box
    ax.add_patch(Rectangle((nx*box[0], ny*box[1]), nx*(box[2]), ny*(box[3]),
                edgecolor = 'blue',
                fill=False,
                lw=1))

    ax.add_patch(Circle((int(mouth_pos[0]*nx), int(mouth_pos[1]*ny)), nx/40.,
                edgecolor = 'purple',
                fill=False,
                lw=1))

    ax.add_patch(Circle((optimal_pos[1], optimal_pos[0]), nx/60.,
                edgecolor = 'green',
                fill=False,
                lw=1))

    ax.add_patch(Rectangle((optimal_pos[1] - 1*int(width * nx), optimal_pos[0] - 1*int(height * ny)), int(width * nx), int(height * ny),
                edgecolor = 'green',
                fill=False,
                lw=1))
    #display plot
    plt.show()
    #plt.matshow(s)
    #plt.plot([0,15], [10,10], color="r")
    #plt.Circle((int(mouth_pos[0]*nx), int(mouth_pos[1]*ny)), 5, color="r")
    #plt.show()
    """
