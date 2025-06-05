import numpy as np
from scipy.linalg import eigh, solve
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix, eye as sparse_eye

import models
from utils import stamp
import utils
from plot import plot

class HessianLLE(models.Neighbourhood):
    """
    Hessian Locally Linear Embedding (Hessian LLE) for dimensionality reduction.

    Parameters
    ----------
    model_args : dict
        Dictionary of model arguments.
    n_neighbors : int
        Number of neighbors to use for reconstruction.
    n_components : int
        Number of dimensions for the embedded space.
    reg : float, optional (default=1e-3)
        Regularization parameter for solving the weight matrix.
    hessian_reg : float, optional (default=1e-3)
        Regularization parameter for Hessian estimation.
    """

    def __init__(self, model_args: dict, n_neighbors: int, n_components: int, reg: float = 1e-3, hessian_reg: float = 1e-3):
        super().__init__(model_args, n_neighbors, n_components)
        self.reg = reg
        self.hessian_reg = hessian_reg
        self.weights_ = None
        self.embedding_ = None
        self.embedding_matrix_M_ = None # The matrix M = (I-W)^T (I-W)

    def _neigh_matrix(self, X):
        return super()._neigh_matrix(X)

    def _fit(self, X: np.ndarray):
        """
        Compute the Hessian LLE embedding.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        """
        n_samples, n_features = X.shape

        # 1. Neighborhood Graph (same as LLE)
        self.neigh_matrix(X) # This computes and sets self.NM

        # 2. Compute Reconstruction Weights (modified to incorporate Hessian)
        self.weights_ = self._compute_hessian_lle_weights(X)

        # 3. Construct M matrix
        I_W = sparse_eye(n_samples) - self.weights_
        M = I_W.T @ I_W
        self.embedding_matrix_M_ = M

        # 4. Solve Eigenvalue Problem
        # try:
        #     eigenvalues, eigenvectors = eigsh(M, k=self.n_components + 1, which='SM')
        #     eigenvalues = np.real(eigenvalues)
        #     eigenvectors = np.real(eigenvectors)
            
        #     # Sort eigenvalues and eigenvectors
        #     sort_indices = np.argsort(eigenvalues)
        #     eigenvalues = eigenvalues[sort_indices]
        #     eigenvectors = eigenvectors[:, sort_indices]

        #     # Discard the first eigenvector (corresponding to the smallest eigenvalue, which is near zero)
        #     self.embedding_ = eigenvectors[:, 1:(self.n_components + 1)]

        #     if self.model_args['verbose']:
        #         print(f"Computed Eigenvalues (smallest {self.n_components + 1}):", eigenvalues[:self.n_components + 1])
        #         print(f"Selected Eigenvalues (indices 1 to {self.n_components}):", eigenvalues[1:(self.n_components + 1)])

        # except Exception as e:
            # print(f"eigsh failed ({e}), attempting dense solver eigh...")
        eigenvalues, eigenvectors = eigh(M.toarray())

        # Sort eigenvalues and eigenvectors
        sort_indices = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[sort_indices]
        eigenvectors = eigenvectors[:, sort_indices]

        # Discard the first eigenvector
        self.embedding_ = eigenvectors[:, 1:(self.n_components + 1)]
        if self.model_args['verbose']:
            print(f"Computed Eigenvalues (smallest {self.n_components + 2}):", eigenvalues[:self.n_components + 2])
            print(f"Selected Eigenvalues (indices 1 to {self.n_components}):", eigenvalues[1:(self.n_components + 1)])

        return self

    def _compute_hessian_lle_weights(self, X: np.ndarray) -> csr_matrix:
        """
        Compute the reconstruction weights for each data point
        using its neighbors, incorporating Hessian information.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        csr_matrix of shape (n_samples, n_samples)
            Reconstruction weights.
        """

        n_samples, n_features = X.shape
        weights = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            neighbors_i = np.where(self.NM[i])[0]  # Indices of neighbors
            X_neighbors = X[neighbors_i] - X[i] # centered neighbors

            # -- Hessian Estimation --
            H_i = self._estimate_hessian(X_neighbors) # (n_neighbors, n_neighbors)

            # -- Modified Weight Computation --
            # Original LLE: minimize ||x_i - \sum_j w_ij x_j||^2
            # Hessian LLE: minimize ||x_i - \sum_j w_ij x_j||^2 + hessian_reg * w_i^T H_i w_i
            #  where w_i is the weight vector for point i

            G = X_neighbors @ X_neighbors.T # (n_neighbors, n_neighbors)
            
            # Add regularization
            G = G + np.eye(X_neighbors.shape[0]) * self.reg
            
            try:
                w_i = solve(G, np.ones(G.shape[0]), assume_a='sym')
                w_i /= w_i.sum() # Normalize weights
            except LinAlgError as e:
                print(f"Singular matrix in weight computation for point {i}: {e}. Using equal weights instead.")
                w_i = np.ones(self.n_neighbors) / self.n_neighbors

            weights[i, neighbors_i] = w_i

        return csr_matrix(weights)

    def _estimate_hessian(self, X_neighbors: np.ndarray) -> np.ndarray:
        """
        Estimate the Hessian matrix for a local neighborhood.

        This is a simplified estimation and more robust methods
        may be needed in practice (e.g., using local quadratic fitting).

        Parameters
        ----------
        X_neighbors : np.ndarray of shape (n_neighbors, n_features)
            The local neighborhood of data points, centered on the
            current point.

        Returns
        -------
        np.ndarray of shape (n_neighbors, n_neighbors)
            Estimated Hessian matrix for the neighborhood.
        """

        n_neighbors, n_features = X_neighbors.shape
        H = np.zeros((n_neighbors, n_neighbors))

        # Simplified Hessian approximation:
        # Using squared distances between neighbors as a proxy
        dist_matrix = utils.intra_dist_matrix(X_neighbors)
        H = dist_matrix + np.eye(n_neighbors) * self.hessian_reg  # Add regularization

        return H

    def _transform(self) -> np.ndarray:
        return self.embedding_
class ENG(models.extensions.ENG, HessianLLE):
    pass