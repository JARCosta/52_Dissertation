# models/lle.py

import numpy as np
from scipy.linalg import eigh, solve, LinAlgError, pinv
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix, eye as sparse_eye

import models
from utils import stamp
from plot import plot
import utils

class LocallyLinearEmbedding(models.Neighbourhood):
    """
    Locally Linear Embedding (LLE) for dimensionality reduction.

    Parameters
    ----------
    model_args : dict
        Dictionary of model arguments, including 'plotation', 'verbose', '#neighs', 'model'.
    n_neighbors : int
        Number of neighbors to use for reconstruction.
    n_components : int
        Number of dimensions for the embedded space.
    reg : float, optional (default=1e-3)
        Regularization parameter for solving the weight matrix.
    """
    def __init__(self, model_args: dict, n_neighbors: int, n_components: int, reg: float = 1e-3):
        super().__init__(model_args, n_neighbors, n_components)
        self.reg = reg
        self.weights_ = None
        self.embedding_matrix_M_ = None # The matrix M = (I-W)^T (I-W)

    def _neigh_matrix(self, X: np.ndarray):
        """
        Computes the k-nearest neighbor graph (non-symmetric).
        LLE uses the direct neighbors for reconstruction.
        """
        # Use the default neigh_matrix which finds k closest neighbors for each point
        neigh_matrix = utils.neigh_matrix(X, self.n_neighbors)
        return neigh_matrix # Returns distances, we only need indices later

    def _fit(self, X: np.ndarray):
        """
        Compute the reconstruction weights and the LLE embedding matrix M.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input data.
        """
        n_samples, n_features = X.shape
        if self.n_neighbors >= n_samples:
             raise ValueError("n_neighbors must be less than n_samples.")
        if self.n_neighbors <= 0:
            raise ValueError("n_neighbors must be positive.")

        stamp.set()

        # 2. Compute Reconstruction Weights W
        W = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            # print(f"{i}/{n_samples}", end="\r")
            neighbors_i = np.where(self.NM[i] != 0)[0].tolist() # Indices of neighbors for point i
            X_i = X[i]
            X_neighbors = X[neighbors_i] # Coordinates of neighbors

            # Center the neighborhood points relative to X_i
            # We want to solve for w in: X_i approx sum(w_j * X_j)
            # Equivalent to minimizing || sum(w_j * (X_j - X_i)) ||^2
            # Let Z = X_neighbors - X_i (shape: n_neighbors, n_features)
            Z = X_neighbors - X_i
            
            # Compute local covariance matrix C = Z @ Z.T (shape: n_neighbors, n_neighbors)
            C = Z @ Z.T
            
            # Add regularization C = C + reg * trace(C) * I
            # trace = np.trace(C)
            # if trace > 0:
            #     R = self.reg * trace
            # else:
            #     R = self.reg # Failsafe for zero trace
            # C += R * np.eye(len(neighbors_i))

            # Alternative regularization (more common): C = C + reg * I
            C = C + self.reg * np.eye(len(neighbors_i))

            # Solve C w = 1 for weights w (vector of size n_neighbors)
            # try:
            # Weights should sum to 1
            # We solve C w = 1 (1 is a vector of ones)
            ones_vec = np.ones(len(neighbors_i))
            # w = solve(C, ones_vec, assume_a='pos') # Use 'pos' if C is positive definite
            w = pinv(C) @ ones_vec # Pseudo-inverse for stability

            # Normalize weights to sum to 1
            w /= np.sum(w)
            
            # Store weights in the main matrix W
            W[i, neighbors_i] = w

            # except LinAlgError:
            #     print(f"Warning: Singular matrix encountered for point {i}. Setting weights to zero.")
            #     print(C)
            #     breakpoint()
            #     # Handle cases where C is singular - could assign equal weights or zero
            #     W[i, neighbors_i] = 1.0 / len(neighbors_i) # Fallback: equal weights
            # except Exception as e:
            #      print(f"Warning: Error solving for weights for point {i}: {e}")
            #      W[i, neighbors_i] = 1.0 / len(neighbors_i) # Fallback

        self.weights_ = csr_matrix(W)
        stamp.print(f"*\t {self.model_args['model']}\t Computed Weights")

        # 3. Compute Embedding Matrix M = (I - W)^T (I - W)
        I = sparse_eye(n_samples)
        M = (I - self.weights_).T @ (I - self.weights_)
        self.embedding_matrix_M_ = M # Store sparse M

        # The base Spectral class sets self.kernel_ for transformation.
        # Here, M is the matrix whose eigenvectors we need.
        self.kernel_ = self.embedding_matrix_M_ # Assign M to kernel_ for transform

        return self

    def _transform(self):
        """
        Perform spectral embedding by finding the bottom eigenvectors of M = (I-W)^T(I-W).
        """
        if self.kernel_ is None: # M matrix stored in kernel_
            raise ValueError("Embedding matrix M is not initialized. Run fit(X) first.")

        # We need the smallest eigenvalues/vectors of M, excluding the constant eigenvector (eigenvalue 0).
        # Use eigsh with which='SM' (Smallest Magnitude or Smallest Algebraic for symmetric M).
        # We need n_components + 1 eigenvectors to discard the first one (corresponding to eigenvalue ~0).
    # try:
    #     # Increase k slightly to improve chances of finding enough non-degenerate eigenvalues
    #     k_request = self.n_components + 2 
        
    #     # M is symmetric positive semi-definite. Use 'SA' for smallest algebraic.
    #     # sigma=0 helps target eigenvalues near zero. tol might need adjustment.
    #     # eigenvalues, eigenvectors = eigsh(self.kernel_, k=k_request, which='SA', sigma=0, tol=1e-9)
    #     eigenvalues, eigenvectors = eigsh(self.kernel_, k=k_request, which='SM', sigma=0)
        
    #     # Sort by eigenvalue magnitude (algebraic should be fine for SA)
    #     sort_indices = np.argsort(eigenvalues)
    #     eigenvalues = eigenvalues[sort_indices]
    #     eigenvectors = eigenvectors[:, sort_indices]

    #     # Discard the eigenvector corresponding to the smallest eigenvalue (~0)
    #     # Indices 1 to n_components+1 give the next n_components eigenvectors
    #     self.embedding_ = eigenvectors[:, 1:(self.n_components + 1)]

    #     if self.model_args['verbose']:
    #         print(f"Computed Eigenvalues (smallest {k_request}):", eigenvalues)
    #         print(f"Selected Eigenvalues (indices 1 to {self.n_components}):", eigenvalues[1:(self.n_components + 1)])

    # except Exception as e:
        # stamp.print(f"*\t {self.model_args['model']}\t Error in eigsh: {e}, attempting dense solver eigh...")
        # Fallback to dense solver
        eigenvalues, eigenvectors = eigh(self.kernel_.toarray()) # eigh sorts eigenvalues

        # Discard the first eigenvector (smallest eigenvalue ~0)
        self.embedding_ = eigenvectors[:, 1:(self.n_components + 1)]

        if self.model_args['verbose']:
                print(f"Computed Eigenvalues (smallest {self.n_components + 2}):", eigenvalues[:self.n_components + 2])
                print(f"Selected Eigenvalues (indices 1 to {self.n_components}):", eigenvalues[1:(self.n_components + 1)])

        if self.model_args['plotation']:
            plot(self.embedding_, title=f"{self.model_args['model']} output", block=False)

class ENG(models.extensions.ENG, LocallyLinearEmbedding):
    pass