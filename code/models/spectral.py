from abc import ABC, abstractmethod
import numpy as np
from scipy.sparse.linalg import eigsh

from utils import stamp
import utils

class Spectral(ABC):
    def __init__(self, model_args:dict, n_components:int|None=None):
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
        
        ret = self._fit(X)
        stamp.print(f"*\t {self.model_args['model']}\t fit")
        return ret

    def _restrict_components(self, eigenvalues:np.ndarray, eigenvectors:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Restrict the number of components to the top `n_components`."""
        
        # Sort eigenvalues and eigenvectors in descending eigenvalue order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:, idx]
        # if self.model_args['verbose']:
        #     print(f"Sorted Eigenvalues:", eigenvalues)

        if self.n_components is None:
            # Compute the top components

            for idx in range(len(eigenvalues)): # Possible because eigenvalues are sorted
                representation = np.sum(eigenvalues[:idx]) / np.sum(eigenvalues)
                if representation > 0.98:
                    break
            
            # idx = 100
            # representation = np.sum(eigenvalues[:idx]) / np.sum(eigenvalues)
            
            stamp.print(f"*\t Smart selected {idx} components ({representation*100:.2f}%)")
            
            if self.model_args['#components'] is None:
                utils.warning(f"There is no information for the intrinsic dimensionality of the data. Selected {idx} components ({representation*100:.2f}%).")

            if idx != self.model_args['#components'] and self.model_args['#components'] is not None:
                utils.warning(f"Count of Smart selected components ({idx}) is different from the estimated intrinsic dimensionality ({self.model_args['#components']})")

            #     eigenvalues_selected = eigenvalues[:idx]
            #     eigenvectors_selected = eigenvectors[:, :idx]
                
            #     idx_restricted = self.model_args['#components']
            #     eigenvalues_restricted = eigenvalues[:idx_restricted]
            #     eigenvectors_restricted = eigenvectors[:, :idx_restricted]

            #     restricted_representation = np.sum(eigenvalues_restricted) / np.sum(eigenvalues)

            #     utils.warning(f"Smart selected {idx} ({representation*100:.2f}%), but {idx_restricted} ({restricted_representation*100:.2f}%) is the intrinsic dimensionality, restricting.")                
            #     if self.model_args['plotation'] and idx_restricted < 3:
            #         embedding_selected = eigenvectors_selected @ np.diag(eigenvalues_selected)  # Project data
            #         embedding_restricted = eigenvectors_restricted @ np.diag(eigenvalues_restricted)  # Project data
                    
            #         NM = self.NM if hasattr(self, 'NM') else np.zeros((1,1))
            #         plot.plot_two(embedding_selected, embedding_restricted, NM, NM, block=False, title=f"{self.model_args['model']} output, and restricted output")
                
            #     idx = idx_restricted
            #     self.model_args['restricted'] = True

            eigenvalues = eigenvalues[:idx]
            eigenvectors = eigenvectors[:, :idx]
            if self.model_args['verbose']:
                print(f"Eigenvalues (selected {idx}):", 100 * eigenvalues / np.sum(eigenvalues))
        else:
            # Take only the top `n_components`
            eigenvalues = eigenvalues[:self.n_components]
            eigenvectors = eigenvectors[:, :self.n_components]
            
            restricted_representation = np.sum(eigenvalues[:self.n_components]) / np.sum(eigenvalues)
            if restricted_representation < 0.98:
                utils.warning(f"The representation is only {restricted_representation*100:.2f}% of the original data.")
            else:
                for idx in range(len(eigenvalues)): # Possible because eigenvalues are sorted
                    representation = np.sum(eigenvalues[:idx]) / np.sum(eigenvalues)
                    if representation > 0.98:
                        break
                utils.warning(f"Selected {self.n_components} ({restricted_representation*100:.2f}%), but {idx} components would already be enough to represent {representation*100:.2f}% of the data.")


            if self.model_args['verbose']:
                print(f"Restricted to {self.n_components} components")
                print(f"Eigenvalues (top {self.n_components}):", eigenvalues)

        return eigenvalues, eigenvectors

    def _transform(self) -> np.ndarray | None:
        if self.kernel_ is None:
            utils.warning("Kernel matrix is not initialized. Run fit(X) first.")
            return
        if np.isnan(self.kernel_).any():
            utils.hard_warning("Kernel matrix contains NaNs. Skipping.")
            return

        if not np.allclose(self.kernel_, self.kernel_.T):
            raise ValueError(f"Kernel matrix is not symmetric. {len(np.where(~np.isclose(self.kernel_, self.kernel_.T, atol=1e-8))[0])} values differ")
        eigenvalues, eigenvectors = np.linalg.eigh(self.kernel_)
        
        eigenvalues, eigenvectors = self._restrict_components(eigenvalues, eigenvectors)

        # eigenvalues = np.sqrt(np.maximum(eigenvalues, 0))  # Ensure non-negative eigenvalues

        embeddings = eigenvectors @ np.diag(eigenvalues ** 0.5)  # Project data

        if hasattr(self, 'recovering_ratio') and self.recovering_ratio is not None:
            
            old_embedding = embeddings.copy()
            embeddings = embeddings * self.recovering_ratio

            if self.model_args['verbose']:
                if callable(getattr(self, '_neigh_matrix', None)):
                    NM_before = self._neigh_matrix(old_embedding)
                else:
                    NM_before = utils.neigh_matrix(old_embedding, self.n_neighbors, bidirectional=True)
                temp = np.where(NM_before > 0)
                print()
                print("NM output:")
                for i in range(-5, 5):
                    print(f"{temp[0][i]} <-> {temp[1][i]} = {NM_before[temp[0][i], temp[1][i]]:.4f}")
                print()
                print(f"recovering ratio: {self.recovering_ratio}")
                print(f"before recovery scalling:")
                print(f"\t min: {np.min(NM_before[NM_before != 0]):.4f} , max: {np.max(NM_before):.4f}")
                print()

            self.recovering_ratio = None

        if self.model_args['verbose'] and callable(getattr(self, '_neigh_matrix', None)):
            NM = self._neigh_matrix(embeddings)
            print(f"Distances after linearized:")
            print(f"\t min: {np.min(NM[NM > 0]):.4f}, max: {np.max(NM):.4f}")
        
        return embeddings
    
    def transform(self):
        """Performs spectral embedding using the top eigenvectors."""
        stamp.set()

        self.embedding_ = self._transform()
        stamp.print(f"*\t {self.model_args['model']}\t transform")
        
        return self.embedding_


    def fit_transform(self, X: np.ndarray):
        """Fits the model and computes embeddings."""
        return self.fit(X).transform()
