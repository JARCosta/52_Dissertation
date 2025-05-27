import numpy as np

import utils
from models import neighbourhood

def TC(X, Y, n_neighbors) -> tuple[float, float]:
    
    n_samples = X.shape[0]
    
    NM_X = neighbourhood.k_neigh(X, n_neighbors)[1] # need symetrix
    NM_Y = neighbourhood.k_neigh(Y, n_neighbors)[1] # need symetrix

    NM_X = np.where(NM_X != 0, 1, 0)
    NM_Y = np.where(NM_Y != 0, 1, 0)

    NM_T = NM_Y - NM_X

    NM_T[np.where(NM_T == -1)] = 0
    D_X = utils.intra_dist_matrix(X)
    R_X = np.argsort(np.argsort(D_X, axis=1), axis=1)
    T = NM_T * R_X
    T[T != 0] -= n_neighbors
    T = 1 - np.sum(T) * (2 / (n_samples * n_neighbors * (2*n_samples - 3*n_neighbors - 1)))



    NM_C = NM_X - NM_Y
    NM_C[np.where(NM_C == -1)] = 0
    D_Y = utils.intra_dist_matrix(Y)
    R_Y = np.argsort(np.argsort(D_Y, axis=1), axis=1)
    C = NM_C * R_Y
    C[C != 0] -= n_neighbors
    C = 1 - np.sum(C) * (2 / (n_samples * n_neighbors * (2*n_samples - 3*n_neighbors - 1)))



    # print(T, sklearn.manifold.trustworthiness(X, Y, n_neighbors=n_neighbors))
    # breakpoint()



    return round(float(T), 3), round(float(C), 3)

def one_NN(Y, labels) -> float:
    if labels is None:
        utils.warning("labels is None")
        return None
    
    NM = neighbourhood.k_neigh(Y, 1)[1] # need symetrix

    Y_labels = np.zeros(labels.shape)
    for i in range(Y.shape[0]):
        Y_labels[i] = labels[np.where(NM[i] != 0)]
    
    one_NN = np.count_nonzero(Y_labels - labels) / labels.shape[0]
    return round(float(one_NN), 3)
