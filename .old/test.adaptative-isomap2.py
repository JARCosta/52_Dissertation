import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import MDS
from scipy.linalg import svd

def neighborhood_contraction_matrix(X, k_max, k_min, d):
    """
    Performs the Neighborhood Contraction (NC) algorithm using an adjacency matrix.

    Args:
        X (numpy.ndarray): Input dataset, shape (n_samples, n_features).
        k_max (int): Maximum neighborhood size.
        k_min (int): Minimum neighborhood size.
        d (int): Intrinsic dimensionality of the manifold.

    Returns:
        numpy.ndarray: Adjacency matrix representing the neighborhoods,
                       float: Maximum r_i_k value encountered
    """
    N = X.shape[0]
    adjacency_matrix = np.zeros((N, N))
    max_r_i_k = -np.inf

    for i in range(N):
        # 1. Initialization
        knn = NearestNeighbors(n_neighbors=k_max+1)
        knn.fit(X)
        distances, indices = knn.kneighbors(X[i].reshape(1, -1))
        indices = indices[0][1:]
        X_i_k = X[indices]  # k-NN neighborhood
        k = k_max

        r_values = np.ones((k_max+1,)) * np.inf  # Initialize r_values for all k
        while True:
            # 2. Compute Singular Values
            X_i_k_centered = X_i_k - np.mean(X_i_k, axis=0)
            U, S, V = svd(X_i_k_centered)
            singular_values = S

            if len(singular_values) <= d:
                raise ValueError("Not enough singular values to compute r_i_k (#neighbours <= d).")
            r_i_k = np.sqrt(np.sum(singular_values[d:]**2) / np.sum(singular_values[:d]**2))
            r_values[k] = r_i_k
            max_r_i_k = max(max_r_i_k, r_i_k)
            
            # 3. Check Criterion
            # The criterion check will now happen outside the contraction loop
            if r_i_k < 0:  # force exit
                adjacency_matrix[i, indices[:k]] = 1
                break

            # 4. Contraction or Termination
            if k > k_min:
                k -= 1
                X_i_k = X[indices[:k]]
            else:
                # 5. Contraction or Termination
                adjacency_matrix[i, indices[:np.argmin(r_values)]] = 1
                break
    return adjacency_matrix, max_r_i_k


def neighborhood_expansion_matrix(X, adjacency_matrix, k_max, eta, d):
    """
    Performs the Neighborhood Expansion (NE) algorithm using an adjacency matrix.

    Args:
        X (numpy.ndarray): Input dataset, shape (n_samples, n_features).
        adjacency_matrix (numpy.ndarray): Adjacency matrix from NC algorithm.
        k_max (int): Maximum neighborhood size.
        eta (float): Threshold for tangent space approximation.
        d (int): Intrinsic dimensionality of the manifold.

    Returns:
        numpy.ndarray: Expanded adjacency matrix.
    """
    N = X.shape[0]
    expanded_adjacency_matrix = adjacency_matrix.copy()

    for i in range(N):
        neighbors_i = np.where(adjacency_matrix[i] == 1)[0]
        X_i = X[neighbors_i]
        x_i_bar = np.mean(X_i, axis=0)

        # 1. Compute Optimal Linear Fitting
        X_i_centered = X_i - x_i_bar
        U, S, Q_i = svd(X_i_centered)
        Q_i = Q_i[:d].T  # Principal components

        # Get the original k_max neighborhood
        knn = NearestNeighbors(n_neighbors=k_max)
        knn.fit(X)
        distances, indices = knn.kneighbors(X[i].reshape(1, -1))
        indices = indices[0][1:]  # Exclude the point itself
        original_neighborhood_indices = indices

        # 2. Compute θ_j^(i) for All Neighbors
        for j in original_neighborhood_indices:
            if j not in neighbors_i:
                x_ij = X[j]
                theta_j_i = Q_i.T @ (x_ij - x_i_bar)

                # 3. Add Suitable Neighbors
                if np.linalg.norm(x_ij - x_i_bar - Q_i @ theta_j_i) <= eta * np.linalg.norm(theta_j_i):
                    expanded_adjacency_matrix[i, j] = 1

    return expanded_adjacency_matrix


def compute_eta_matrix(X, adjacency_matrix, d):
    """
    Adaptively computes the parameter eta based on the adjacency matrix.

    Args:
        X (numpy.ndarray): Input dataset, shape (n_samples, n_features).
        adjacency_matrix (numpy.ndarray): Adjacency matrix representing the neighborhoods.
        d (int): Intrinsic dimensionality of the manifold.

    Returns:
        float: Computed value for eta.
    """
    N = X.shape[0]
    rho_values = []

    for i in range(N):
        neighbors_i = np.where(adjacency_matrix[i] == 1)[0]
        X_i = X[neighbors_i]
        X_i_centered = X_i - np.mean(X_i, axis=0)
        U, S, V = np.linalg.svd(X_i_centered)
        singular_values = S

        if len(singular_values) <= d:
            raise ValueError("Not enough singular values to compute rho_i (#neighbours <= d).")
        rho_i = np.sqrt(np.sum(singular_values[d:]**2) / np.sum(singular_values[:d]**2))
        rho_values.append(rho_i)

    # Sort rho values in decreasing order
    rho_values.sort(reverse=True)

    max_gap = 0
    j = 0

    for i in range(N - 1):
        gap = rho_values[i + 1] / rho_values[i]
        if gap > max_gap:
            max_gap = gap
            j = i

    eta = (rho_values[j] + rho_values[j + 1]) / 2
    return eta


def isomap_adaptive_neighborhood_matrix(X, k_max, k_min, d):
    """
    Isomap with adaptive neighborhood selection using an adjacency matrix.

    Args:
        X (numpy.ndarray): Input dataset, shape (n_samples, n_features).
        k_max (int): Maximum neighborhood size.
        k_min (int): Minimum neighborhood size.
        d (int): Intrinsic dimensionality of the manifold.

    Returns:
        numpy.ndarray: Low-dimensional embedding.
    """

    # 1. Adaptive Neighborhood Construction
    adjacency_matrix, max_r_i_k = neighborhood_contraction_matrix(X, k_max, k_min, d)
    print([np.count_nonzero(neigh) for neigh in adjacency_matrix])
    eta = compute_eta_matrix(X, adjacency_matrix, d)
    print(f"Computed Eta: {eta}, Max r_i_k: {max_r_i_k}")
    adaptive_adjacency_matrix = neighborhood_expansion_matrix(X, adjacency_matrix, k_max, eta, d)
    print([np.count_nonzero(neigh) for neigh in adaptive_adjacency_matrix])
    print(adaptive_adjacency_matrix)

    # 2. Geodesic Distance Computation
    N = X.shape[0]
    dist_matrix = np.full((N, N), np.inf)

    for i in range(N):
        dist_matrix[i, i] = 0
        neighbors_i = np.where(adaptive_adjacency_matrix[i] == 1)[0]
        for j in neighbors_i:
            dist_matrix[i, j] = np.linalg.norm(X[i] - X[j])
            dist_matrix[j, i] = dist_matrix[i, j]  # Ensure symmetry

    # Floyd-Warshall algorithm to compute all-pairs shortest paths
    for k in range(N):
        print(f"{k}/{N}", end="\r")
        for i in range(N):
            for j in range(N):
                dist_matrix[i, j] = min(dist_matrix[i, j], dist_matrix[i, k] + dist_matrix[k, j])

    # 3. Low-Dimensional Embedding
    mds = MDS(n_components=d, dissimilarity='precomputed')
    embedding = mds.fit_transform(dist_matrix)

    return embedding, adaptive_adjacency_matrix


# Generate S-curve data
X, color = make_swiss_roll(n_samples=500, random_state=0)

# Parameters for adaptive Isomap
k_max = 20
k_min = 5
d = 2  # Intrinsic dimensionality of the S-curve

# Apply adaptive Isomap
embedding, adaptive_adjacency_matrix = isomap_adaptive_neighborhood_matrix(X, k_max, k_min, d)

# Visualize the results
fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122)

# Plot original S-curve
ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
ax1.set_title("Original S-curve with Adaptive Neighborhoods")

# Plot embedded S-curve
ax2.scatter(embedding[:, 0], embedding[:, 1], c=color, cmap=plt.cm.Spectral)
ax2.set_title("Embedded S-curve with Corresponding Neighborhoods")

# Plot connections in original space
for i in range(X.shape[0]):
    neighbors_i = np.where(adaptive_adjacency_matrix[i] == 1)[0]
    for j in neighbors_i:
        ax1.plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]], [X[i, 2], X[j, 2]],
                 color='gray', linestyle='-', linewidth=0.5)

# Plot connections in embedded space
for i in range(embedding.shape[0]):
    neighbors_i = np.where(adaptive_adjacency_matrix[i] == 1)[0]
    for j in neighbors_i:
        ax2.plot([embedding[i, 0], embedding[j, 0]], [embedding[i, 1], embedding[j, 1]],
                 color='gray', linestyle='-', linewidth=0.5)

plt.show()