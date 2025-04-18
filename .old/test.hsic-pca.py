import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from scipy.linalg import eigh

def hsic_pca(X, sigma=1.0, n_components=2):
    """ Compute PCA using HSIC instead of covariance """
    n = X.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n  # Centering matrix

    # Compute Kernel Matrix (RBF Kernel)
    K = rbf_kernel(X, X, gamma=1 / (2 * sigma**2))

    # Compute Centered HSIC Matrix
    Kc = H @ K @ H  # Center K
    C_hsic = Kc @ Kc.T  # HSIC-based covariance-like matrix

    # Eigen-decomposition
    eigenvalues, eigenvectors = eigh(C_hsic)

    # Select top components (largest eigenvalues)
    top_indices = np.argsort(-eigenvalues)[:n_components]  # Sort in descending order
    components = eigenvectors[:, top_indices]

    print(X.shape, components.shape)

    # Transform data
    X_hsic_pca = X @ components  # Project data

    return X_hsic_pca, components

# Example Usage
X = np.random.rand(100, 5)  # 100 samples, 5D features
X_transformed, components = hsic_pca(X, sigma=1.0, n_components=2)
print("Transformed Shape:", X_transformed.shape)
