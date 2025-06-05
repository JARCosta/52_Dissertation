import numpy as np
import matlab.engine
import os


eng = None

os.environ['LOKY_MAX_CPU_COUNT'] = str(os.cpu_count())
os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())


############################################################
# DATASETS #################################################
############################################################

import numpy as np

def swiss(n_points, noise, random_state=None):
    np.random.seed(random_state) if random_state is not None else None

    t = (3 * np.pi / 2) * (1 + 2 * np.random.rand(n_points, 1))
    height = 30 * np.random.rand(n_points, 1)
    X = np.hstack([
        t * np.cos(t),
        height,
        t * np.sin(t)
    ]) + noise * np.random.randn(n_points, 3)

    labels = np.remainder(np.sum(np.hstack([np.round(t / 2), np.round(height / 12)]), axis=1), 2)
    t = np.hstack([t, height])

    return X, labels, t


def twinpeaks(n_points, noise, random_state=None):
    np.random.seed(random_state) if random_state is not None else None
    
    xy = 1 - 2 * np.random.rand(2, n_points)
    X = np.hstack([xy.T, (np.sin(np.pi * xy[0, :]) * np.tanh(3 * xy[1, :]))[:, None]]) + noise * np.random.randn(n_points, 3)
    X[:, 2] *= 10
    labels = np.remainder(np.sum(np.round((X - np.min(X, axis=0)) / 10), axis=1), 2)

    return X, labels, None

def helix(n_points, noise, random_state=None):
    np.random.seed(random_state) if random_state is not None else None

    t = np.linspace(1, n_points, n_points)[:, None] / n_points
    t = (t ** 1.0) * 2 * np.pi
    X = np.hstack([(2 + np.cos(8 * t)) * np.cos(t), (2 + np.cos(8 * t)) * np.sin(t), np.sin(8 * t)]) + noise * np.random.randn(n_points, 3)
    labels = np.remainder(np.round(t * 1.5), 2)

    print(X[:10])
    return X, labels, t

############################################################
# UTILS ####################################################
############################################################
from scipy.sparse import csr_matrix

def k_neigh(data, n_neighbors, bidirectional=False):
    """
    Builds a k-nearest neighbor (k-NN) neighborhood matrix.

    Args:
        data (numpy.ndarray): The dataset (n_samples, n_features).
        bidirectional (bool, optional): If True, the connections are bidirectional, neighbourhood matrix is upper triangular.
            Defaults to False, neighbourhood matrix is directly computed from the k-nn graph.
        common_neighbors (bool, optional): If True, considers nodes with common neighbours to be neighbours as well.
            Defaults to False.

    Returns:
        numpy.ndarray: The neighborhood matrix (n_samples, n_samples).
    """
    def intra_dist_matrix(data: np.ndarray):
        from sklearn.neighbors import NearestNeighbors
        
        n_samples = data.shape[0]
        dist, indices = NearestNeighbors(n_neighbors=n_samples).fit(data).kneighbors(data)
        dist_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            dist_matrix[i][indices[i]] = dist[i]
        return dist_matrix

    k = n_neighbors
    n_samples = data.shape[0]
    dist_matrix = intra_dist_matrix(data[:, :5])
    neigh_matrix = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        nearest_indices = np.argsort(dist_matrix[i])[0:k + 1]
        
        nearest_indices = np.delete(nearest_indices, np.where(nearest_indices == i))
        nearest_indices = nearest_indices[0:k]

        for j in nearest_indices:
            neigh_matrix[i, j] = dist_matrix[i, j] if dist_matrix[i, j] > 0 else 1e-10

    print(f"connections: {np.count_nonzero(neigh_matrix)}")
    if bidirectional: # remove duplicate connections
        neigh_bidirectional = np.maximum(neigh_matrix, neigh_matrix.T)
        neigh_matrix = np.triu(neigh_bidirectional)

    print(f"connections: {np.count_nonzero(neigh_matrix)}")
    return neigh_matrix

def k_graph(neigh_matrix:np.ndarray):
    return csr_matrix(neigh_matrix != 0)


############################################################
# MVU ######################################################
############################################################



class MVU:

    def __init__(self, n_neighbors:int=5):
        global eng
        if eng is None:
            eng = matlab.engine.start_matlab()
            # Add the directory containing the MATLAB function to the MATLAB path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            eng.addpath(current_dir, nargout=0)
        self.n_neighbors = n_neighbors


    def _fit(self, X:np.ndarray):
        """Fit the MVU model and compute the low-dimensional embeddings using MATLAB."""
        self.NM = k_neigh(X, self.n_neighbors, bidirectional=True)
        print(f"connections: {np.count_nonzero(self.NM)}")
        # Convert data to MATLAB format
        X_matlab = matlab.double(X.tolist())
        
        # NM is a nxn sparse matrix of neighbourhood connections (distance(i,j) where j is in the neighbourhood of i)
        rows, cols = k_graph(self.NM).nonzero()
        
        neighbor_pairs = [[int(i) + 1, int(j) + 1] for i, j in zip(rows, cols)]

        N_matlab = matlab.double(neighbor_pairs)
        
        K_matlab, cvx_status = eng.solve_mvu_optimization(X_matlab, N_matlab, nargout=2)
        self.kernel_ = np.array(K_matlab, dtype=np.float64)
        return self


    def _transform(self):
        """Transform the data to the low-dimensional space."""
        eigenvalues, eigenvectors = np.linalg.eigh(self.kernel_)

        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:, idx]
        print(f"Sorted eigenvalues: {eigenvalues[:10]}")

        eigenvalues_idx = []
        for i in idx:
            if eigenvalues[i] > (np.median(eigenvalues) + np.std(eigenvalues)):
                eigenvalues_idx.append(i)
        
        eigenvalues = eigenvalues[eigenvalues_idx]
        eigenvectors = eigenvectors[:, eigenvalues_idx]

        print(f"Selected eigenvalues: {eigenvalues}")
        embedding = eigenvectors @ np.diag(eigenvalues)
        return embedding


    def fit_transform(self, X:np.ndarray):
        """Fit the model and transform the data to the low-dimensional space."""
        return self._fit(X)._transform()




if __name__ == "__main__":
    # X, labels, t = swiss(1000, 0.1, 11)
    # X, labels, t = twinpeaks(1000, 0.1, 11)
    X, labels, t = helix(1000, 0.1, 11)
    print(f"Loaded data")
    mvu = MVU(n_neighbors=5)
    print("Initialized MVU")
    embedding = mvu.fit_transform(X)
    print(embedding.shape)