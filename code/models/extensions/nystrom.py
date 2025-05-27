import numpy as np
import json

from scipy.sparse import csr_matrix, csgraph
from scipy.linalg import pinv
from scipy.spatial.distance import cdist

from plot import plot
import measure
import models
from utils import stamp



class Nystrom:
    def __init__(self, ratio:float=None, subset_indices:list=None):
        if ratio is None and subset_indices is None:
            raise ValueError("Either 'ratio' or 'subset_indices' must be provided.")
        self.subset_indices = subset_indices
        self.ratio = ratio
        
    def _fit(self, X):
        self.n_samples = X.shape[0]
        if self.subset_indices is None:
            print(f"Randomly selecting {int(self.n_samples * self.ratio)} samples from {self.n_samples} samples.")
            self.subset_indices = np.random.choice(self.n_samples, int(self.n_samples * self.ratio), replace=False)
        self.remaining_indices = np.setdiff1d(np.arange(self.n_samples), self.subset_indices)

        self.X = X
        self.X_subset = X[self.subset_indices]

        self.neigh_matrix(self.X_subset) # TODO: handle non NG methods
        super()._fit(self.X_subset)

        if self.model_args['plotation']:
            super()._transform()
            # print(f"Subset 1-NN: {measure.one_NN(self.embedding_, self.subset_indices)}")
            # print(f"Subset T, C: {measure.TC(self.X_subset, self.embedding_, self.model_args['#neighs'])}")
            plot(self.embedding_, block=False, title="Subset Nystrom", legend=json.dumps(self.model_args, indent=2))
        
        D_remaining_subset = cdist(self.X, self.X_subset)
        # self.neigh_matrix(self.X)
        # D_remaining_subset = csgraph.shortest_path(self.NM, directed=False, method='D') ** 2  # Squared distance matrix
        # D_remaining_subset = D_remaining_subset[:, self.subset_indices]
        
        # Convert distances to a kernel matrix (Gaussian-like transformation)
        gamma = 1  # You'll need to tune this parameter
        K_remaining_subset = np.exp(-gamma * D_remaining_subset**2)
        
        # Compute nystrom embeddings
        self.kernel_ = K_remaining_subset @ pinv(self.kernel_) @ K_remaining_subset.T
        return self

    def transform(self):
        """Performs spectral embedding using the top eigenvectors."""
        t = super().transform()
        if self.model_args['plotation']:
            colors = np.zeros(self.n_samples)
            colors[self.subset_indices] = 1
            plot(self.embedding_, block=False, c=colors, title="Final Nystrom", legend=json.dumps(self.model_args, indent=2))
        return t

    def fit_transform(self, X):
        return models.spectral.Spectral.fit_transform(self, X)
