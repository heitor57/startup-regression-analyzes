import numpy as np

def gauss_seidel(A,b,error=0.0001):
    x_old = np.zeros(b.shape[0])
    max_dist = np.inf
    while max_dist>error:
        x_new = np.zeros(b.shape[0])
        for i in range(A.shape[0]):
            x_new[i] = b[i][0]
            for j in range(A.shape[1]):# A.nrow == A.ncol
                if i != j:
                    if j < i:
                        x_new[i] -= A[i][j]*x_new[j]
                    else:
                        x_new[i] -= A[i][j]*x_old[j]
            x_new[i] /= A[i][i]

        max_dist = 0
        for i in range(b.shape[0]):
            max_dist = max(max_dist,abs(x_new[i] - x_old[i]))
        x_old = x_new
    return x_new
