from abc import ABC, abstractmethod
import numpy as np
from scipy.sparse.linalg import eigsh

from plot import plot
from utils import save_cache
from utils import stamp

class Spectral(ABC):
    def __init__(self, model_args:dict, n_components:int=None):
        self.model_args = model_args
        self.n_components = n_components
        self.embedding_ = None
        self.kernel_ = None

    @abstractmethod
    def _fit(self, X: np.ndarray):
        return self

    def fit(self, X: np.ndarray):
        """Computes the kernel matrix."""
        stamp.set()
        
        print("Fitting Spectral...")
        ret = self._fit(X)
        # save_cache(self.model_args, self.kernel_, "K")
        
        stamp.print(f"*\t {self.model_args['model']}\t fit")
        return ret

    def _transform(self, verbose:bool=False) -> np.ndarray | None:
        if self.kernel_ is None:
            # raise ValueError("Kernel matrix is not initialized. Run fit(X) first.")
            print("Warning: Kernel matrix is not initialized. Run fit(X) first.")
            return
        
        if self.n_components is None:
            eigenvalues, eigenvectors = np.linalg.eigh(self.kernel_)
        else:
            eigenvalues, eigenvectors = eigsh(self.kernel_, k=self.n_components, which='LM')
        
        # Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:, idx]
        # print(f"Sorted Eigenvalues:", eigenvalues)

        if self.n_components is None:
            # Compute the top components
            if verbose:
                print("std:", np.std(eigenvalues))
                print("median:", np.median(eigenvalues))

            eigenvalues_idx = []
            for i in idx:
                if eigenvalues[i] > (np.median(eigenvalues) + np.std(eigenvalues)):
                    eigenvalues_idx.append(i)

            eigenvalues = eigenvalues[:self.model_args['#components']:]
            eigenvectors = eigenvectors[:, :self.model_args['#components']:]
            if verbose:
                print(f"Eigenvalues (top {self.model_args['#components']}):", eigenvalues)
        else:
            # Take only the top `n_components`
            eigenvalues = eigenvalues[:self.n_components]
            eigenvectors = eigenvectors[:, :self.n_components]
            
            if verbose:
                print(f"Eigenvalues (top {self.n_components}):", eigenvalues)

        # eigenvalues = np.sqrt(np.maximum(eigenvalues, 0))  # Ensure non-negative eigenvalues

        self.embedding_ = eigenvectors @ np.diag(eigenvalues)  # Project data

        return self.embedding_

    def transform(self):
        """Performs spectral embedding using the top eigenvectors."""
        stamp.set()

        self._transform()
        # save_cache(self.model_args, self.embedding_, "Y")
        
        stamp.print(f"*\t {self.model_args['model']}\t transform")
        
        return self.embedding_


    def fit_transform(self, X: np.ndarray):
        """Fits the model and computes embeddings."""
        return self.fit(X).transform()
