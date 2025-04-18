import numpy as np
import scipy
from sklearn.datasets import make_swiss_roll

from utils import connected as fully_connected




def adaptative_ng(X):

    def connections_to_ng(connections:np.ndarray):
        n = connections.shape[0]
        NM = np.zeros((n, n))
        NM[np.where(connections != 0)] = 1
        return NM

    def connect(D_X:np.ndarray, connections:np.ndarray, connections_counter:np.ndarray, n_neighbours:int):
        connected_count = 1
        while np.amin(connections_counter) < n_neighbours:

            min = np.argmin(D_X)
            a = min // n
            b = min % n

            connections[a][b] = connected_count
            connected_count += 1

            try:
                connections_counter[a] += 1
            except KeyError:
                connections_counter[a] = 1
            try:
                connections_counter[b] += 1
            except KeyError:
                connections_counter[b] = 1

            D_X[a][b] = np.inf

    def disconnect(connections:np.ndarray, connections_counter:np.ndarray, n_neighbours:int):
        while True:
            a = np.argmax([np.sum(np.count_nonzero(i)) for i in connections])
            b = np.argmax(connections[a])
            # print(a, b)

            if connections_counter[a]-1 > n_neighbours and connections_counter[b]-1 > n_neighbours:
                connections_counter[a] -= 1
                connections_counter[b] -= 1
                connections[a][b] = 0
            else:
                break




    n = X.shape[0]
    connections = np.zeros((n,n))
    connections_counter = np.zeros((n, 1))
    D_X = scipy.spatial.distance_matrix(X, X) + np.array(np.diag([np.inf, ] * n))

    n_neighbours = 0
    while not fully_connected(connections_to_ng(connections)):
        n_neighbours += 1
        # print(n_neighbours)
        connect(D_X, connections, connections_counter, n_neighbours)






    while fully_connected(connections_to_ng(connections)):
        previous_connections = connections.copy()
        n_neighbours -= 1
        # print(n_neighbours)
        disconnect(connections, connections_counter, n_neighbours)
    connections = previous_connections



    NM = connections_to_ng(connections)

    # remove duplicate connections
    for i in range(NM.shape[0]):
        for j in range(NM.shape[1]):
            if NM[i][j] != 0:
                NM[j][i] = 0


    # print(connections_counter)
    # print(connections)
    # print([list(i) for i in NM])
    # print(np.count_nonzero(NM))

    return NM


if __name__ == "__main__":
    X, _ = make_swiss_roll(n_samples=1000, noise=0.0, random_state=10)
    NM = adaptative_ng(X)

    from plot import plot
    plot(X, NM)



