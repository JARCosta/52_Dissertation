import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.neighbors import NearestNeighbors

# Keep the functions defined previously
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import MDS
from scipy.linalg import svd

def neighborhood_contraction(X, k_max, k_min, eta, d):
    """
    Performs the Neighborhood Contraction (NC) algorithm.

    Args:
        X (numpy.ndarray): Input dataset, shape (n_samples, n_features).
        k_max (int): Maximum neighborhood size.
        k_min (int): Minimum neighborhood size.
        eta (float): Threshold for tangent space approximation.
        d (int): Intrinsic dimensionality of the manifold.

    Returns:
        list: List of adaptively selected neighborhoods for each point.
    """
    N = X.shape[0]
    neighborhoods = []

    for i in range(N):
        # 1. Initialization
        knn = NearestNeighbors(n_neighbors=k_max)
        knn.fit(X)
        distances, indices = knn.kneighbors(X[i].reshape(1, -1))
        X_i_k = X[indices[0]]  # k-NN neighborhood
        k = k_max

        while True:
            # 2. Compute Singular Values
            X_i_k_centered = X_i_k - np.mean(X_i_k, axis=0)
            U, S, V = svd(X_i_k_centered)
            singular_values = S

            r_i_k = np.sqrt(np.sum(singular_values[d:]**2) / np.sum(singular_values[:d]**2))

            # 3. Check Criterion
            if r_i_k < eta:
                neighborhoods.append(indices[0][:k])
                break

            # 4. Contraction or Termination
            if k > k_min:
                k -= 1
                X_i_k = X[indices[0][:k]]
            else:
                r_values = []
                for k_temp in range(k_min, k_max + 1):
                    X_i_k_temp = X[indices[0][:k_temp]]
                    X_i_k_temp_centered = X_i_k_temp - np.mean(X_i_k_temp, axis=0)
                    U_temp, S_temp, V_temp = svd(X_i_k_temp_centered)
                    singular_values_temp = S_temp
                    r_i_k_temp = np.sqrt(np.sum(singular_values_temp[d:]**2) / np.sum(singular_values_temp[:d]**2))
                    r_values.append(r_i_k_temp)

                k_i = k_min + np.argmin(r_values)
                neighborhoods.append(indices[0][:k_i])
                break
    return neighborhoods

def neighborhood_expansion(X, neighborhoods, k_max, eta, d):
    """
    Performs the Neighborhood Expansion (NE) algorithm.

    Args:
        X (numpy.ndarray): Input dataset, shape (n_samples, n_features).
        neighborhoods (list): List of neighborhoods obtained from NC algorithm.
        k_max (int): Maximum neighborhood size.
        eta (float): Threshold for tangent space approximation.
        d (int): Intrinsic dimensionality of the manifold.

    Returns:
        list: List of expanded neighborhoods.
    """

    N = X.shape[0]
    expanded_neighborhoods = []

    for i in range(N):
        X_i = X[neighborhoods[i]]
        x_i_bar = np.mean(X_i, axis=0)

        # 1. Compute Optimal Linear Fitting
        X_i_centered = X_i - x_i_bar
        U, S, Q_i = svd(X_i_centered)
        Q_i = Q_i[:d].T  # Principal components

        # Get the original k_max neighborhood
        knn = NearestNeighbors(n_neighbors=k_max)
        knn.fit(X)
        distances, indices = knn.kneighbors(X[i].reshape(1, -1))
        original_neighborhood_indices = indices[0]

        expanded_neighborhood = neighborhoods[i].tolist()

        # 2. Compute θ_j^(i) for All Neighbors
        for j in original_neighborhood_indices:
            if j not in expanded_neighborhood:
                x_ij = X[j]
                theta_j_i = Q_i.T @ (x_ij - x_i_bar)

                # 3. Add Suitable Neighbors
                if np.linalg.norm(x_ij - x_i_bar - Q_i @ theta_j_i) <= eta * np.linalg.norm(theta_j_i):
                    expanded_neighborhood.append(j)

        expanded_neighborhoods.append(np.array(expanded_neighborhood))
    return expanded_neighborhoods

def isomap_adaptive_neighborhood(X, k_max, k_min, eta, d):
    """
    Isomap with adaptive neighborhood selection.

    Args:
        X (numpy.ndarray): Input dataset, shape (n_samples, n_features).
        k_max (int): Maximum neighborhood size.
        k_min (int): Minimum neighborhood size.
        eta (float): Threshold for tangent space approximation.
        d (int): Intrinsic dimensionality of the manifold.

    Returns:
        numpy.ndarray: Low-dimensional embedding.
    """

    # 1. Adaptive Neighborhood Construction
    neighborhoods = neighborhood_contraction(X, k_max, k_min, eta, d)
    print([len(neighborhood) for neighborhood in neighborhoods])  # Print neighborhood sizes
    adaptive_neighborhoods = neighborhood_expansion(X, neighborhoods, k_max, eta, d)
    print([len(neighborhood) for neighborhood in adaptive_neighborhoods])  # Print expanded neighborhood sizes

    # 2. Geodesic Distance Computation
    N = X.shape[0]
    dist_matrix = np.full((N, N), np.inf)
    
    for i in range(N):
        dist_matrix[i, i] = 0
        for j in adaptive_neighborhoods[i]:
            dist_matrix[i, j] = np.linalg.norm(X[i] - X[j])
            dist_matrix[j, i] = dist_matrix[i, j]  # Ensure symmetry
    
    # Floyd-Warshall algorithm to compute all-pairs shortest paths
    for k in range(N):
        for i in range(N):
            for j in range(N):
                dist_matrix[i, j] = min(dist_matrix[i, j], dist_matrix[i, k] + dist_matrix[k, j])

    # 3. Low-Dimensional Embedding
    mds = MDS(n_components=d, dissimilarity='precomputed')
    embedding = mds.fit_transform(dist_matrix)

    return embedding, adaptive_neighborhoods  # Return neighborhoods

# Generate S-curve data
X, color = make_swiss_roll(n_samples=500, random_state=0)

# Parameters for adaptive Isomap
k_max = 20
k_min = 5
eta = 0.1
d = 2  # Intrinsic dimensionality of the S-curve

# Apply adaptive Isomap
embedding, adaptive_neighborhoods = isomap_adaptive_neighborhood(X, k_max, k_min, eta, d) # Get neighborhoods

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
    for j in adaptive_neighborhoods[i]:
        ax1.plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]], [X[i, 2], X[j, 2]], 
                 color='gray', linestyle='-', linewidth=0.5)

# Plot connections in embedded space
for i in range(embedding.shape[0]):
    for j in adaptive_neighborhoods[i]:
        ax2.plot([embedding[i, 0], embedding[j, 0]], [embedding[i, 1], embedding[j, 1]], 
                 color='gray', linestyle='-', linewidth=0.5)

plt.show()