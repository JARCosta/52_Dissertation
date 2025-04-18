import numpy as np
from sklearn.neighbors import NearestNeighbors

def estimate_intrinsic_dim_MLE(X, k=10):
    """
    Estimates the intrinsic dimension of a dataset using the Maximum Likelihood Estimation (MLE) method.
    
    Parameters:
        X: ndarray of shape (n_samples, n_features)
            Input data.
        k: int
            Number of nearest neighbors to consider.

    Returns:
        float: Estimated intrinsic dimensionality.
    """
    n_samples = X.shape[0]

    # Compute nearest neighbors
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    distances, _ = nn.kneighbors(X)
    distances = distances[:, 1:]  # Remove self-distance

    # Apply MLE formula to each point
    log_ratios = np.log(distances[:, -1][:, np.newaxis] / distances[:, :-1])
    intrinsic_dims = (1 / np.mean(log_ratios, axis=1))

    # Return the global estimated intrinsic dimension
    return np.mean(intrinsic_dims)

# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_swiss_roll

    X, _ = make_swiss_roll(n_samples=1000, noise=0.1)
    estimated_dim = estimate_intrinsic_dim_MLE(X, k=10)
    print(f"Estimated Intrinsic Dimensionality: {estimated_dim:.2f}")
