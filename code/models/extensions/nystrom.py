import numpy as np
import json

from scipy.linalg import pinv
from scipy.spatial.distance import cdist

from plot import plot
import models
from utils import stamp_set, stamp_print



class Nystrom:
    def __init__(self, ratio:float=None, subset_indices:list=None):
        if ratio is None and subset_indices is None:
            raise ValueError("Either 'ratio' or 'subset_indices' must be provided.")
        self.subset_indices = subset_indices
        self.ratio = ratio
        
    def _fit(self, X):
        self.n_samples = X.shape[0]
        if self.subset_indices is None:
            self.subset_indices = np.random.choice(self.n_samples, int(self.n_samples * self.ratio), replace=False)
        self.remaining_indices = np.setdiff1d(np.arange(self.n_samples), self.subset_indices)
        self.X = X

        self.X_subset = X[self.subset_indices]

        self.neigh_matrix(self.X_subset) # TODO: handle non NG methods
        super()._fit(self.X_subset)

        if self.model_args['plotation']:
            super()._transform()
            plot(self.embedding_, block=False, title="Subset Nystrom", legend=json.dumps(self.model_args, indent=2))
        
        D_remaining_subset = cdist(self.X, self.X_subset)

        # Convert distances to a kernel matrix (Gaussian-like transformation)
        gamma = 0.5  # You'll need to tune this parameter
        K_remaining_subset = np.exp(-gamma * D_remaining_subset**2)
        
        # Compute nystrom embeddings
        self.kernel_ = K_remaining_subset @ pinv(self.kernel_) @ K_remaining_subset.T
        return self

    def transform(self):
        """Performs spectral embedding using the top eigenvectors."""
        stamp_set()

        self._transform()
        # save_cache(self.model_args, self.embedding_, "Y")
        
        stamp_print(f"*\t {self.model_args['model']}\t transform")
        
        if self.model_args['plotation']:
            colors = np.zeros(self.n_samples)
            colors[self.subset_indices] = 1
            plot(self.embedding_, block=False, c=colors, title="Final Nystrom", legend=json.dumps(self.model_args, indent=2))

        return self.embedding_

    def fit_transform(self, X):
        return models.spectral.Spectral.fit_transform(self, X)
