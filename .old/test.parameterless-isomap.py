import numpy as np
import scipy.spatial.distance as dist
import networkx as nx
from sklearn.manifold import MDS
from sklearn.neighbors import NearestNeighbors

from utils import k_neigh
from plot import plot

def parameterless_isomap(X, n_components=2, k_prime=10, lambda_scale=1.0):
    """
    Implements Parameterless Isomap with Adaptive Neighborhood Selection.

    Parameters:
        X (numpy array): (n_samples, n_features) input data.
        n_components (int): Dimension of the low-dimensional embedding.
        k_prime (int): Number of neighbors for local scale estimation.
        lambda_scale (float): Scaling factor for adaptive radius.

    Returns:
        Y (numpy array): (n_samples, n_components) low-dimensional embedding.
    """
    
    # Step 1: Compute local scale (sigma_i)
    nbrs = NearestNeighbors(n_neighbors=k_prime+1).fit(X)
    distances, indices = nbrs.kneighbors(X)  # Get k_prime nearest neighbors

    print(distances, distances.shape)
    # distances[:, 1:] just removes the first neigh of each X_i, the distance to itself
    sigma = np.mean(distances[:, 1:], axis=1)  # for point X_i, compute the mean distance to its neighbours


    # Step 2: Compute adaptive radius (r_i)
    r = lambda_scale * sigma

    # Step 3: Construct adaptive neighborhood graph
    n_samples = X.shape[0]
    G = nx.Graph()
    NM = np.zeros((n_samples, n_samples))
    
    for i in range(n_samples):
        G.add_node(i)
        for j in range(n_samples):
            if i != j:
                d_ij = np.linalg.norm(X[i] - X[j])               # euclidean distance between i and j
                if d_ij <= max(r[i], r[j]): # Adaptive neighborhood condition;           If euclidean(i, j) <= mean distance from i or j to its neighbours
                    G.add_edge(i, j, weight=d_ij)
                    NM[i][j] = 1



    print(np.sum(NM, axis=0))
    print(np.count_nonzero(NM))

    # Step 4: Compute geodesic distance matrix
    D = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        shortest_paths = nx.single_source_dijkstra_path_length(G, i)
        for j in range(n_samples):
            D[i, j] = shortest_paths.get(j, np.inf)  # Inf if disconnected




    # Step 5: Perform Multidimensional Scaling (MDS)
    mds = MDS(n_components=n_components, dissimilarity='precomputed', random_state=42)
    Y = mds.fit_transform(D)

    plot(Y, NM)

    return Y



import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll

# Generate synthetic Swiss Roll dataset
X, _ = make_swiss_roll(n_samples=500, noise=0.1)

NM = k_neigh(X, 6)[1]
print(np.count_nonzero(NM))
plot(X, NM, block=False)

# Apply Parameterless Isomap
Y = parameterless_isomap(X, n_components=2)

# Plot the 2D embedding
plt.scatter(Y[:, 0], Y[:, 1], c=np.arange(len(Y)), cmap='viridis')
plt.colorbar(label='Point Index')
plt.title("Parameterless Isomap Embedding")
plt.show()
