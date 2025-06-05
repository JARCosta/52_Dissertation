from abc import ABC, abstractmethod
import numpy as np
from scipy.sparse.linalg import eigsh

import plot
from utils import stamp, warning

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

    def _transform(self) -> np.ndarray | None:
        if self.kernel_ is None:
            warning("Kernel matrix is not initialized. Run fit(X) first.")
            return
        
        if self.n_components is None:
            eigenvalues, eigenvectors = np.linalg.eigh(self.kernel_)
        else:
            eigenvalues, eigenvectors = eigsh(self.kernel_, k=self.n_components, which='LM')
        
        # Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:, idx]
        if self.model_args['verbose']:
            print(f"Sorted Eigenvalues:", eigenvalues)

        if self.n_components is None:
            # Compute the top components
            if self.model_args['verbose']:
                print("std:", np.std(eigenvalues))
                print("median:", np.median(eigenvalues))

            eigenvalues_idx, representation = [], 0
            for eig_idx in range(len(eigenvalues)):
                eigenvalues_idx.append(eig_idx)
                representation += eigenvalues[eig_idx] / np.sum(eigenvalues)
                if representation > 0.98:
                    break
            
            stamp.print(f"*\t Smart selected {len(eigenvalues_idx)} components ({representation*100:.2f}%)")

            if len(eigenvalues_idx) != self.model_args['#components']:
                warning(f"Smart eigenvalue selection found {len(eigenvalues_idx)} components, but {self.model_args['#components']} components were requested. Forcing the selection of {self.model_args['#components']} components.")
                eigenvalues = eigenvalues[eigenvalues_idx]
                eigenvectors = eigenvectors[:, eigenvalues_idx]
                embedding = eigenvectors @ np.diag(eigenvalues)  # Project data
                
                eigenvalues_idx = range(self.model_args['#components'])

                eigenvalues = eigenvalues[eigenvalues_idx]
                eigenvectors = eigenvectors[:, eigenvalues_idx]
                embedding_restricted = eigenvectors @ np.diag(eigenvalues)  # Project data
                
                plot.plot_two(embedding, embedding_restricted, title=f"{self.model_args['model']} Eigenvalues")

            eigenvalues = eigenvalues[eigenvalues_idx]
            eigenvectors = eigenvectors[:, eigenvalues_idx]
            if self.model_args['verbose']:
                print(f"Eigenvalues (selected {len(eigenvalues_idx)}):", eigenvalues)
        else:
            # Take only the top `n_components`
            eigenvalues = eigenvalues[:self.n_components]
            eigenvectors = eigenvectors[:, :self.n_components]
            
            if self.model_args['verbose']:
                print(f"Eigenvalues (top {self.n_components}):", eigenvalues)

        # eigenvalues = np.sqrt(np.maximum(eigenvalues, 0))  # Ensure non-negative eigenvalues

        self.embedding_ = eigenvectors @ np.diag(eigenvalues)  # Project data

        return self.embedding_

    def transform(self):
        """Performs spectral embedding using the top eigenvectors."""
        stamp.set()

        self._transform()
        stamp.print(f"*\t {self.model_args['model']}\t transform")
        
        return self.embedding_


    def fit_transform(self, X: np.ndarray):
        """Fits the model and computes embeddings."""
        return self.fit(X).transform()
