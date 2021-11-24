import numpy as np

def integrateMatrix(m):
    s = np.zeros(m.shape)

    # sum line by line from left to right
    for i in range(m.shape[0]):
        prev = 0
        for j in range(m.shape[1]):
            prev += m[i,j]
            s[i,j] = prev

    for j in range(m.shape[1]):
        prev = 0
        for i in range(m.shape[0]):
            prev += s[i,j]
            s[i,j] = prev

    return s

