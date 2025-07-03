from abc import ABC, abstractmethod
import numpy as np
from scipy.sparse.linalg import eigsh

import plot
from utils import stamp
import utils

class Spectral(ABC):
    def __init__(self, model_args:dict, n_components:int=None):
        self.model_args = model_args
        self.n_components = n_components
        self.embedding_ = None
        self.kernel_ = None
        # self.plot = plt.subplots(2, 2, figsize=(12, 12), subplot_kw={'projection': '3d'})

    @abstractmethod
    def _fit(self, X: np.ndarray):
        return self

    def fit(self, X: np.ndarray):
        """Computes the kernel matrix."""
        stamp.set()
        
        print("Fitting Spectral...")
        ret = self._fit(X)
        stamp.print(f"*\t {self.model_args['model']}\t fit")
        return ret

    def _restrict_components(self, eigenvalues:np.ndarray, eigenvectors:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Restrict the number of components to the top `n_components`."""
        
        # Sort eigenvalues and eigenvectors in descending eigenvalue order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:, idx]
        if self.model_args['verbose']:
            print(f"Sorted Eigenvalues:", eigenvalues)

        if self.n_components is None:
            # Compute the top components

            for idx in range(len(eigenvalues)): # Possible because eigenvalues are sorted
                representation = np.sum(eigenvalues[:idx]) / np.sum(eigenvalues)
                if representation > 0.98:
                    break
            
            stamp.print(f"*\t Smart selected {idx} components ({representation*100:.2f}%)")
            
            if self.model_args['#components'] is None:
                utils.warning(f"There is no information for the intrinsic dimensionality of the data. Selected {idx} components ({representation*100:.2f}%).")

            if idx != self.model_args['#components'] and self.model_args['#components'] is not None:
                
                eigenvalues_selected = eigenvalues[:idx]
                eigenvectors_selected = eigenvectors[:, :idx]
                
                idx_restricted = self.model_args['#components']
                eigenvalues_restricted = eigenvalues[:idx_restricted]
                eigenvectors_restricted = eigenvectors[:, :idx_restricted]

                restricted_representation = np.sum(eigenvalues_restricted) / np.sum(eigenvalues)
                utils.warning(f"The theoretical number of components is {idx_restricted}, but {idx} components were selected ({representation*100:.2f}%). Reducing representation to {restricted_representation*100:.2f}%.")
                
                if self.model_args['plotation']:
                    embedding_selected = eigenvectors_selected @ np.diag(eigenvalues_selected)  # Project data
                    embedding_restricted = eigenvectors_restricted @ np.diag(eigenvalues_restricted)  # Project data
                    plot.plot_two(embedding_selected, embedding_restricted, self.NM, self.NM, block=False, title=f"{self.model_args['model']} output, and restricted output")
                
                idx = idx_restricted
                self.model_args['restricted'] = True

            eigenvalues = eigenvalues[:idx]
            eigenvectors = eigenvectors[:, :idx]
            if self.model_args['verbose']:
                print(f"Eigenvalues (selected {idx}):", eigenvalues)
        else:
            # Take only the top `n_components`
            eigenvalues = eigenvalues[:self.n_components]
            eigenvectors = eigenvectors[:, :self.n_components]
            
            if self.model_args['verbose']:
                print(f"Restricted to {self.n_components} components")
                print(f"Eigenvalues (top {self.n_components}):", eigenvalues)

        return eigenvalues, eigenvectors

    def _transform(self) -> np.ndarray | None:
        if self.kernel_ is None:
            utils.warning("Kernel matrix is not initialized. Run fit(X) first.")
            return
        if np.isnan(self.kernel_).any():
            utils.warning("Kernel matrix contains NaNs. Skipping.")
            return

        if not np.allclose(self.kernel_, self.kernel_.T):
            raise ValueError(f"Kernel matrix is not symmetric. {len(np.where(~np.isclose(self.kernel_, self.kernel_.T, atol=1e-8))[0])} values differ")
        eigenvalues, eigenvectors = np.linalg.eigh(self.kernel_)
        
        eigenvalues, eigenvectors = self._restrict_components(eigenvalues, eigenvectors)

        # eigenvalues = np.sqrt(np.maximum(eigenvalues, 0))  # Ensure non-negative eigenvalues

        return eigenvectors @ np.diag(eigenvalues)  # Project data

    def transform(self):
        """Performs spectral embedding using the top eigenvectors."""
        stamp.set()

        self.embedding_ = self._transform()
        stamp.print(f"*\t {self.model_args['model']}\t transform")
        
        return self.embedding_


    def fit_transform(self, X: np.ndarray):
        """Fits the model and computes embeddings."""
        return self.fit(X).transform()
