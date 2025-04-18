
import numpy as np


def inter_connections(X:np.ndarray):
    """
    C <- -1_{|X|x|X|}

    for i = 1 to |X| do
        for j = 1 to |X| do
            if i != j do
                r <- 1/|X_j| (X_j.T * 1_d)
                C_{ij} <- argmin diag (X_i - r) @ (X_i - r).T
    """

    C = np.zeros((len(X), len(X))) - 1
    
    for i in range(len(X)):
        for j in range(len(X)):
            if i != j:
                r = X[j].mean(0)
                closest_to_j_mean_coords = X[i][np.argmin(np.diag((X[i] - r) @ (X[i] - r).T))]
                C[i][j] = np.where(np.all(X[i] == closest_to_j_mean_coords, axis=1))[0]

    return C
