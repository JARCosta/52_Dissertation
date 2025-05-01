from abc import abstractmethod
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import warnings


from models.spectral import Spectral
from plot import plot
from utils import stamp
import utils

class Neighbourhood(Spectral):
    def __init__(self, model_args:dict, n_neighbors:int, n_components:int=None):
        super().__init__(model_args, n_components)
        self.n_neighbors = n_neighbors

    def k_neigh(self, data, bidirectional=False, common_neighbors=False):
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
        k = self.n_neighbors
        n_samples = data.shape[0]
        dist_matrix = utils.intra_dist_matrix(data) # TODO: I don't get how can sklearn.neighbors.NearestNeighbors be so much faster than scipy.spatial.distance.cdist, it knows its cdist between the same data?
        neigh_matrix = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            # Find indices of k nearest neighbors (excluding itself)
            nearest_indices = np.argsort(dist_matrix[i])[1:k + 1]
            for j in nearest_indices:
                neigh_matrix[i, j] = dist_matrix[i, j]
        if bidirectional:
            neigh_bidirectional = np.maximum(neigh_matrix, neigh_matrix.T) # connections are unidirectional
            neigh_matrix = np.triu(neigh_bidirectional) # keep upper triangle.

        if common_neighbors:
            adj_bool = neigh_matrix > 0
            common_neigh = adj_bool.astype(int) @ adj_bool.astype(int).T
            common_neigh = (common_neigh > 0) & (~adj_bool) # selection of common neighbors that are not already connected
            for i,j in zip(*np.where(common_neigh)):
                shortest_path = np.inf
                # Find the k node that minimizes the distance between i and j
                for k in range(n_samples):
                    if neigh_matrix[i,k] > 0 and neigh_matrix[j,k] > 0:
                        shortest_path = min(shortest_path, neigh_matrix[i,k] + neigh_matrix[j,k])
                neigh_matrix[i,j] = shortest_path
                neigh_matrix[j,i] = shortest_path
        
        neigh_graph = csr_matrix(neigh_matrix != 0)
        return neigh_graph, neigh_matrix

    def neigh_matrix(self, X):
        stamp.set()
        self.NM = self._neigh_matrix(X)
        stamp.print(f"*\t {self.model_args['model']}\t neigh_matrix\t {np.count_nonzero(self.NM)} connections")
        if self.model_args['plotation']:
            plot(X, self.NM, title=f"{self.model_args['model']} neigh_matrix, k={self.n_neighbors}", block=False)
        return self.NM

    @abstractmethod
    def _neigh_matrix(self, X:np.ndarray):
        return self.k_neigh(X)[1]
    
    def fit_transform(self, X: np.ndarray):
        """Fits the model and computes embeddings."""
        self.neigh_matrix(X)
        return self.fit(X).transform()
