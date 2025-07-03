# models/laplacian.py

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh, pinv

import models
from utils import stamp
from plot import plot
import utils

class LaplacianEigenmaps(models.Neighbourhood):
    """
    Laplacian Eigenmaps for dimensionality reduction.

    Parameters
    ----------
    model_args : dict
        Dictionary of model arguments, including 'plotation', 'verbose', '#neighs', 'model'.
    n_neighbors : int
        Number of neighbors to use for building the adjacency graph.
    n_components : int
        Number of dimensions for the embedded space.
    sigma : float, optional (default=1.0)
        Width parameter for the weight matrix.
    """
    def __init__(self, model_args: dict, n_neighbors: int, n_components: int, sigma: float = 1.0):
        super().__init__(model_args, n_neighbors, n_components)
        self.sigma = sigma
        self.laplacian_ = None
        self.degree_matrix_ = None

    def _neigh_matrix(self, X: np.ndarray):
        """
        Computes the k-nearest neighbor graph and returns the distance matrix.
        Uses bidirectional connections.
        """
        neigh_matrix = utils.neigh_matrix(X, self.n_neighbors, bidirectional=True)
        # neigh_matrix = neigh_matrix + neigh_matrix.T # no need for symmetry here?, Weight matrix will be made symmetric
        return neigh_matrix

    def _fit(self, X: np.ndarray):
        """
        Compute the graph Laplacian matrix.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input data.
        """
        n_samples = X.shape[0]
        
        # 1. Compute Adjacency Graph (Weight Matrix W)
        neigh_dist_matrix = self.neigh_matrix(X) # Get distances for neighbors
        
        W = np.zeros((n_samples, n_samples))
        non_zero_indices = np.where(neigh_dist_matrix > 0)

        # Gaussian kernel function: W_ij = exp(-||x_i - x_j||^2 / (2 * sigma^2))
        dists_sq = neigh_dist_matrix[non_zero_indices]**2
        self.sigma = np.std(dists_sq)
        W[non_zero_indices] = np.exp(-dists_sq / (2 * self.sigma**2))
        # Make W symmetric
        W = np.maximum(W, W.T)
        self.kernel_ = W # Store the weight matrix if needed, though LE uses Laplacian

        # 2. Compute Degree Matrix D
        D = np.diag(np.sum(W, axis=1))
        self.degree_matrix_ = D # Store degree matrix for generalized eigenproblem

        # 3. Compute Graph Laplacian L = D - W
        L = D - W
        self.laplacian_ = csr_matrix(L) # Store sparse Laplacian

        # Note: The base Spectral class assumes self.kernel_ is the matrix
        # for the standard eigenvalue problem. For LE, we need L v = lambda D v.
        # We will handle this in the _transform method.
        # Setting self.kernel_ = L might work if D is identity, but not generally.
        # We'll store L and D separately.

        return self

    def _transform(self):
        """
        Perform spectral embedding using the eigenvectors of the generalized
        eigenvalue problem L v = lambda D v.
        """
        if self.laplacian_ is None or self.degree_matrix_ is None:
            raise ValueError("Laplacian matrix is not initialized. Run fit(X) first.")

        stamp.set()
        print(f"Solving generalized eigenvalue problem: L v = lambda D v")

        # We need the smallest non-zero eigenvalues/vectors.
        # eigsh finds eigenvalues with the largest magnitude by default (which='LM').
        # We need the smallest algebraic values ('SM').
        # We ask for n_components + 1 because the smallest eigenvalue is often 0
        # corresponding to the constant eigenvector, which we discard.
        # try:
        #     # Use eigsh for sparse matrices
        #     # Ensure D is sparse for eigsh compatibility if L is sparse
        #     D_sparse = diags(np.diag(self.degree_matrix_), shape=self.degree_matrix_.shape, format='csr')

        #     # Increase k slightly to improve chances of finding enough non-degenerate eigenvalues
        #     k_request = self.n_components + 2 
            
        #     # Use shift-invert mode for smallest eigenvalues near sigma=0
        #     # Note: M=D defines the generalized problem L v = lambda D v
        #     eigenvalues, eigenvectors = eigsh(self.laplacian_, k=k_request, M=D_sparse, 
        #                                       which='SM', sigma=0, tol=1e-6) 
                                              
        #     # eigsh with sigma=0 might return eigenvalues near 0. Sort them algebraically.
        #     sort_indices = np.argsort(eigenvalues)
        #     eigenvalues = eigenvalues[sort_indices]
        #     eigenvectors = eigenvectors[:, sort_indices]

        #     # Discard the first eigenvector (corresponding to eigenvalue ~0)
        #     self.embedding_ = eigenvectors[:, 1:(self.n_components + 1)]

        #     if self.model_args['verbose']:
        #          print(f"Computed Eigenvalues (smallest {k_request}):", eigenvalues)
        #          print(f"Selected Eigenvalues (indices 1 to {self.n_components}):", eigenvalues[1:(self.n_components + 1)])


        # except Exception as e:
        #      print(f"eigsh failed ({e}), attempting dense solver eigh...")
        
        # Fallback to dense solver if sparse fails (might be slow/memory intensive)
        # Need to handle potential singularity in D if using standard transform L_rw = D^-1 L
        # Using eigh directly on (L, D) handles generalized problem
        eigenvalues, eigenvectors = eigh(self.laplacian_.toarray(), self.degree_matrix_)
        
        # Sort eigenvalues and select smallest non-zero ones
        sort_indices = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[sort_indices]
        eigenvectors = eigenvectors[:, sort_indices]

        # Discard the first eigenvector (corresponding to eigenvalue ~0)
        self.embedding_ = eigenvectors[:, 1:(self.n_components + 1)]

        if self.model_args['verbose']:
            print(f"Computed Eigenvalues (smallest {self.n_components + 1}):", eigenvalues[:self.n_components + 2])
            print(f"Selected Eigenvalues (indices 1 to {self.n_components}):", eigenvalues[1:(self.n_components + 1)])

        stamp.print(f"*\t {self.model_args['model']}\t transform (Laplacian)")

        if self.model_args['plotation']:
            plot(self.embedding_, title=f"{self.model_args['model']} output", block=False)

class ENG(models.extensions.ENG, LaplacianEigenmaps):
    pass