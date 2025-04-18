import numpy as np
from models.spectral import Spectral

class PCA(Spectral):
    def __init__(self, model_args, n_components = None):
        super().__init__(model_args, n_components)

    def _fit(self, X: np.ndarray):
        """Computes the kernel matrix (default: similarity matrix)."""
        self.kernel_ = X @ X.T
        return self
