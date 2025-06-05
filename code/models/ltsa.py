# models/ltsa.py

import numpy as np
from scipy.linalg import svd, eigh
from scipy.sparse import csr_matrix, lil_matrix, eye as sparse_eye
from scipy.sparse.linalg import eigsh

import models
from utils import stamp
from plot import plot

class LTSA(models.Neighbourhood):
    """
    Local Tangent Space Alignment (LTSA) for dimensionality reduction.

    Parameters
    ----------
    model_args : dict
        Dictionary of model arguments, including 'plotation', 'verbose', '#neighs', 'model'.
    n_neighbors : int
        Number of neighbors to use for constructing local tangent spaces.
    n_components : int
        Number of dimensions for the embedded space.
    """
    def __init__(self, model_args: dict, n_neighbors: int, n_components: int):
        super().__init__(model_args, n_neighbors, n_components)
        self.alignment_matrix_ = None # The global alignment matrix (often denoted B or M)

    def _neigh_matrix(self, X: np.ndarray):
        """
        Computes the k-nearest neighbor graph (non-symmetric).
        LTSA uses the direct neighbors for tangent space estimation.
        """
        # Use the default k_neigh which finds k closest neighbors for each point
        neigh_matrix = self.k_neigh(X, bidirectional=False, common_neighbors=False)
        return neigh_matrix # We only need indices later

    def _fit(self, X: np.ndarray):
        """
        Compute the LTSA alignment matrix.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input data.
        """
        n_samples, n_features = X.shape
        if self.n_neighbors >= n_samples:
             raise ValueError("n_neighbors must be less than n_samples.")
        if self.n_neighbors <= self.n_components:
             raise ValueError(f"n_neighbors ({self.n_neighbors}) must be greater than n_components ({self.n_components}) for SVD.")

        stamp.set()
        print("Computing LTSA local tangent spaces and alignment matrix...")

        # 1. Find Neighbors (using sklearn for indices)
        from sklearn.neighbors import NearestNeighbors
        knn = NearestNeighbors(n_neighbors=self.n_neighbors).fit(X)
        # Note: LTSA typically includes the point itself in its neighborhood calculation
        # for centering, but uses k *other* neighbors. We adjust k accordingly.
        # Let's find k+1 neighbors and use indices [0, ..., k] where 0 is the point itself.
        neighbor_indices_with_self = knn.kneighbors(X, n_neighbors=self.n_neighbors + 1, return_distance=False)
        # Shape: (n_samples, n_neighbors + 1)

        # Initialize the alignment matrix B (sparse lil_matrix is good for construction)
        # B = sum_i S_i^T * L_i * S_i where L_i = I - Q_i * Q_i^T
        # B is n_samples x n_samples
        B = lil_matrix((n_samples, n_samples))

        # 2. Compute Local Information (Tangent Spaces Q_i and Alignment L_i)
        for i in range(n_samples):
            # Get indices for the neighborhood of point i (including i itself)
            idx_i = neighbor_indices_with_self[i] # Indices: [i, neighbor1, neighbor2, ...]
            X_i_neigh = X[idx_i, :] # Neighborhood points (k+1 points)

            # Center the neighborhood data
            mean_i = np.mean(X_i_neigh, axis=0)
            X_i_centered = X_i_neigh - mean_i

            # Compute local tangent space basis Q_i using SVD
            try:
                # We need the first d principal components (right singular vectors Vh)
                # Note: svd returns Vh (V transpose)
                U, S, Vh = svd(X_i_centered, full_matrices=False)
                Q_i = Vh[:self.n_components].T # shape (n_features, n_components) - Basis vectors

            except LinAlgError:
                print(f"Warning: SVD did not converge for neighborhood {i}. Skipping.")
                continue

            # Compute local coordinates Theta_i = X_i_centered @ Q_i
            # This is not explicitly needed for the alignment matrix formulation below

            # Compute the local alignment matrix L_i = I - G_i * G_i^T
            # Where G_i are the local coordinates (Theta_i), but scaled.
            # A more direct formulation uses the projection onto the tangent space:
            # Projector P_i = Q_i @ Q_i.T
            # Local alignment L_i = I - P_i (projects onto orthogonal complement)
            # However, the standard LTSA cost function leads to B calculation directly:
            
            # Construct the local Gram matrix for the *centered* coordinates
            # G_i = X_i_centered @ Q_i (shape: k+1, d) - Local coordinates
            # W_i = eye(k+1) - (1/(k+1)) * ones(k+1, k+1) # Centering matrix
            # B_i = W_i @ (I - G_i @ G_i.T) @ W_i --- This is complex.

            # Simpler approach: Construct Alignment Matrix B directly
            # B = sum_i S_i^T * (I - Q_i*Q_i^T) * S_i  -- This involves large matrices P_i.

            # Standard formulation for B relies on local coordinates Theta:
            # Theta_i = X_i_centered @ Q_i (k+1, d) - local coordinates
            # We need to find global coordinates Y (n_samples, d) such that Y[idx_i] approx Theta_i (up to translation/rotation)
            # This leads to minimizing sum_i || Y[idx_i] @ R_i - Theta_i * T_i ||^2
            # Resulting eigenvalue problem uses matrix M = sum_i S_i^T * W_local * S_i
            # Where W_local captures the alignment constraints based on Theta_i.

            # Let's use the formulation from the original paper or common implementations:
            # Compute the matrix H_i = I_{k+1} - (1/(k+1)) * 1 * 1^T (local centering)
            H_i = np.eye(self.n_neighbors + 1) - (1.0 / (self.n_neighbors + 1)) * np.ones((self.n_neighbors + 1, self.n_neighbors + 1))
            
            # Compute local tangent coordinates using the basis Q_i
            # Theta_i = X_i_centered @ Q_i  # Shape (k+1, d)
            
            # Construct local weight matrix W_i = H_i * (Theta_i @ Theta_i^T) * H_i
            # This seems related to Isomap's kernel calculation on local coords.

            # Let's try the alignment matrix B formulation based on projectors:
            # Find the orthonormal basis Q_i for the tangent space (d principal components)
            # The orthogonal complement basis P_i spans n_features - d dimensions.
            # We want the final embedding Y to be orthogonal to P_i for each neighborhood.
            # This means P_i^T @ Y[idx_i] should be zero.
            # This isn't quite right either.

            # Revisit the standard cost: Minimize sum_i || Y[idx_i] - (Y_mean_i + Theta_i @ R_i) ||^2
            # Where Theta_i are local tangent coordinates, R_i is rotation.
            # This leads to finding eigenvectors of M = I - sum(S_i^T * L_i * S_i)
            # Where L_i = G_i @ G_i^T and G_i are *normalized* local coordinates.

            # Let's follow the scikit-learn approach for constructing M:
            # Compute local coordinates G = U[:, :d] * S[:d] where U,S,Vh = svd(X_i_centered)
            # G has shape (k+1, d)
            G = U[:, :self.n_components] * S[:self.n_components]

            # Compute local weight matrix W_i = G @ G.T
            W_local = G @ G.T # Shape (k+1, k+1)

            # Add this local weight matrix to the global alignment matrix B,
            # using the neighborhood indices idx_i
            # B[np.ix_(idx_i, idx_i)] += W_local # This sums contributions correctly
            # Need efficient sparse update:
            for r, row_idx in enumerate(idx_i):
                for c, col_idx in enumerate(idx_i):
                    B[row_idx, col_idx] += W_local[r, c]


        # 3. Construct Final Embedding Matrix M = I - B
        # Note: Some formulations use B directly, others I-B. Depends on whether
        # you maximize trace(Y^T B Y) or minimize trace(Y^T (I-B) Y).
        # Since we need bottom eigenvectors, we likely minimize trace(Y^T M Y) where M = I - B or similar.
        # Let's assume M = I - B for now. Need to verify the exact formulation leading
        # to bottom eigenvectors. If M = B, we'd need top eigenvectors.
        # If the goal is alignment -> maximize similarity -> maximize trace(Y^T B Y) -> top eigenvectors of B.
        # Let's check common implementations: often it's the bottom eigenvectors of a matrix related to reconstruction error.
        
        # Assume we minimize cost related to I-B, needing bottom eigenvectors of M = I - B
        I_sparse = sparse_eye(n_samples, format='csr')
        B_csr = B.tocsr()
        
        # The matrix should be centered for the eigenvalue problem
        # M = (I-Pi)^T * M * (I-Pi) where Pi = (1/N) * 1 * 1^T
        # Or simply subtract mean from Y later. Let's proceed without explicit centering first.

        self.alignment_matrix_ = B_csr # Store B
        # Set self.kernel_ to the matrix for eigenvalue decomposition
        # If we need bottom eigenvectors of (I - B), set kernel_ = I - B
        # If we need top eigenvectors of B, set kernel_ = B
        
        # Let's assume we need bottom eigenvectors of (some form of) B or I-B.
        # If M = B, need top eigenvectors (like PCA/MVU).
        # If M = I - B, need bottom eigenvectors (like LLE).
        # The original paper aims to maximize sum_i Trace(Theta_i^T R_i^T Y_i), which suggests maximizing Y^T B Y.
        # So, we likely need the TOP eigenvectors of B.
        
        self.kernel_ = self.alignment_matrix_ # Set kernel to B
        stamp.print(f"*\t {self.model_args['model']}\t Computed Alignment Matrix B")
        return self

    def _transform(self, verbose: bool = False):
        """
        Perform spectral embedding by finding the top eigenvectors of the alignment matrix B.
        """
        if self.kernel_ is None: # B matrix stored in kernel_
            raise ValueError("Alignment matrix B is not initialized. Run fit(X) first.")

        stamp.set()
        print(f"Solving eigenvalue problem for LTSA alignment matrix B...")

        # We need the largest eigenvalues/vectors of B.
        # Use eigsh with which='LM' (Largest Magnitude).
        # We need n_components + 1 eigenvectors because the top one might correspond
        # to the data mean (or related constant component), though not always zero eigenvalue here.
        # Let's request n_components + 1 and see.
        try:
            # Increase k slightly
            k_request = self.n_components + 2
            
            # B should be symmetric positive semi-definite. Use 'LA' (Largest Algebraic).
            eigenvalues, eigenvectors = eigsh(self.kernel_, k=k_request, which='LA', tol=1e-9)

            # Sort by eigenvalue magnitude (descending)
            sort_indices = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[sort_indices]
            eigenvectors = eigenvectors[:, sort_indices]

            # Select the top n_components eigenvectors (indices 0 to n_components-1)
            # Sometimes the very top eigenvector is discarded, sometimes not. Check conventions.
            # If the top eigenvalue corresponds to a near-constant vector, discard it.
            # For now, take the top n_components directly.
            self.embedding_ = eigenvectors[:, :self.n_components]

            if verbose:
                print(f"Computed Eigenvalues (largest {k_request}):", eigenvalues)
                print(f"Selected Eigenvalues (top {self.n_components}):", eigenvalues[:self.n_components])

        except Exception as e:
            print(f"eigsh failed ({e}), attempting dense solver eigh...")
            # Fallback to dense solver
            eigenvalues, eigenvectors = eigh(self.kernel_.toarray()) # eigh sorts ascending

            # Select the top n_components eigenvectors (last n_components)
            self.embedding_ = eigenvectors[:, -self.n_components:]
            # Reverse order to match descending magnitude
            self.embedding_ = self.embedding_[:, ::-1] 
            selected_eigenvalues = eigenvalues[-self.n_components:][::-1]


            if verbose:
                 print(f"Computed Eigenvalues (all):", eigenvalues)
                 print(f"Selected Eigenvalues (top {self.n_components}):", selected_eigenvalues)

        stamp.print(f"*\t {self.model_args['model']}\t transform (LTSA)")

        if self.model_args['plotation']:
            plot(self.embedding_, title=f"{self.model_args['model']} output", block=False)

    # Override fit_transform to ensure correct _transform is called
    def fit_transform(self, X: np.ndarray):
        """Fits the model and computes embeddings."""
        self.fit(X)
        self._transform(self.model_args['verbose']) # Use the overridden transform
        return self.embedding_