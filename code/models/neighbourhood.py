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

    def neigh_matrix(self, X):
        stamp.set()
        self.NM = self._neigh_matrix(X)
        stamp.print(f"*\t {self.model_args['model']}\t neigh_matrix\t {np.count_nonzero(self.NM)} connections")
        return self.NM

    @abstractmethod
    def _neigh_matrix(self, X:np.ndarray):
        return utils.neigh_matrix(X, self.n_neighbors)
    
    def fit_transform(self, X: np.ndarray):
        """Fits the model and computes embeddings."""
        self.neigh_matrix(X)
        self.fit(X).transform()
        return self.embedding_