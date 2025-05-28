from abc import abstractmethod
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import warnings


from models.spectral import Spectral
from plot import plot
from utils import stamp, k_neigh

class Neighbourhood(Spectral):
    def __init__(self, model_args:dict, n_neighbors:int, n_components:int=None):
        super().__init__(model_args, n_components)
        self.n_neighbors = n_neighbors

    def k_neigh(self, X:np.ndarray, bidirectional:bool=False, common_neighbors:bool=False):
        return k_neigh(X, self.n_neighbors, bidirectional, common_neighbors)

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
