import numpy as np
from scipy.spatial import distance_matrix
from scipy.sparse.linalg import eigsh
from numpy.linalg import eig, eigh

class LocallyLinearEmbedding:
    def __init__(self, n_neighbors=5, n_components=2, reg=1e-3):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.reg = reg  # Regularization term

    def _compute_weights(self, X):
        """Computes the reconstruction weights for each point using its k-nearest neighbors."""
        n_samples = X.shape[0]
        W = np.zeros((n_samples, n_samples))

        dist_matrix = distance_matrix(X, X)
        for i in range(n_samples):
            neighbors = np.argsort(dist_matrix[i])[1:self.n_neighbors+1]
            Z = X[neighbors] - X[i]  # Centered neighbors
            C = Z @ Z.T  # Local covariance

            # Regularization (in case of singular matrix)
            C += np.eye(self.n_neighbors) * self.reg * np.trace(C)

            # Solve for reconstruction weights
            w = np.linalg.solve(C, np.ones(self.n_neighbors))
            w /= w.sum()  # Normalize weights

            # Store weights
            W[i, neighbors] = w

        return W

    def _compute_embedding(self, W):
        """Computes the low-dimensional embedding using the bottom eigenvectors of (I - W).T (I - W)."""
        n_samples = W.shape[0]
        M = np.eye(n_samples) - W
        M = M.T @ M  # Compute (I - W)^T (I - W)

        print(M)
        breakpoint()

        # Solve the eigenproblem: find n_components smallest eigenvectors
        eigvals, eigvecs = eigsh(M, k=self.n_components+1, which='SM')

        return eigvecs[:, 1:self.n_components+1]  # Ignore the first eigenvector

    def fit_transform(self, X):
        """Computes the LLE embedding for the input data."""
        W = self._compute_weights(X)
        return self._compute_embedding(W)

# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_swiss_roll
    import matplotlib.pyplot as plt

    # Generate Swiss Roll dataset
    X, _ = make_swiss_roll(n_samples=1000, noise=0.1)
    X = X[:, [0, 2]]  # Consider a 2D projection for visualization

    # Apply LLE
    lle = LocallyLinearEmbedding(n_neighbors=10, n_components=2)
    X_embedded = lle.fit_transform(X)


    from plot import plot
    plot(X_embedded)
    breakpoint()

    # Plot the result
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=np.arange(len(X_embedded)), cmap='Spectral')
    plt.title("Locally Linear Embedding Projection")
    plt.colorbar()
    plt.show()
