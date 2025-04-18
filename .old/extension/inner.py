import numpy as np


def inner_connections(S:np.ndarray, C:np.ndarray):
    """
    R <- {}
    
    for i=0 to |S| do
        A <- 0_|S_i|

        for j=0 to |C_i| do
            if C_{ij} != -1 do
                A <- A + log diag (S_i - C_ij)(S_i - C_ij).T

        for j=|C_i| to d+1 do
            r <- S_{i argmax(A)}
            R_i <- R_i U {r}
            A <- A + log diag (S_i - r) (S_i - r).T
    """

    R = [[] for _ in range(len(S))]

    for i in range(len(S)):
        A = np.zeros(len(S[i]))

        # Consider the already used inter-connection points
        for j in range(len(C[i])):
            if C[i][j] != -1:
                with np.errstate(divide='ignore'):
                    idx = int(C[i][j])
                    A += np.log(np.diag((S[i] - S[i][idx]) @ (S[i] - S[i][idx]).T))


        # Get the rest reference points from inner-connections
        for j in range(len(C[i])-1, len(S[i][0])+1):
            r_idx = np.argmax(A)
            R[i].append(r_idx)
            with np.errstate(divide='ignore'):
                A += np.log(np.diag((S[i] - S[i][r_idx]) @ (S[i] - S[i][r_idx]).T))
    return R
