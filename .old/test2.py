import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import shortest_path
from sklearn.manifold import MDS

from plot import plot
from utils import k_neigh

class AdaptiveIsomap:
    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit_transform(self, X):
        self.X = X
        self.n_samples = X.shape[0]

        # Step 1: Estimate intrinsic dimensionality
        intrinsic_dim = self.estimate_intrinsic_dim(X)
        # print(intrinsic_dim)

        # Step 2: Compute adaptive neighborhoods
        adjacency_matrix = self.compute_adaptive_neighbors(X, intrinsic_dim)

        # Step 3: Compute shortest paths to estimate geodesic distances
        geodesic_distances = shortest_path(adjacency_matrix, directed=False)

        # Step 4: Apply MDS for final embedding
        embedding = self.apply_mds(geodesic_distances)

        return embedding

    def estimate_intrinsic_dim(self, X, k=10):
        """Estimate intrinsic dimensionality using MLE method."""
        
        nn = NearestNeighbors(n_neighbors=k+1).fit(X)
        distances, _ = nn.kneighbors(X)
        distances = distances[:, 1:]  # Exclude self-distance

        log_ratios = np.log(distances[:, -1][:, np.newaxis] / distances[:, :-1])
        intrinsic_dims = (1 / (np.mean(log_ratios, axis=1)))

        return int(np.round(np.mean(intrinsic_dims)))

    def compute_adaptive_neighbors(self, X, intrinsic_dim):
        """Adaptive neighborhood selection based on tangent space consistency."""
        n_samples = X.shape[0]
        adjacency_matrix = np.full((n_samples, n_samples), np.inf)

        NM = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            neighbors = self.get_adaptive_neighbors(X, i, intrinsic_dim)
            # print(len(neighbors))
            for j in neighbors:
                adjacency_matrix[i, j] = np.linalg.norm(X[i] - X[j])
                NM[i][j] = 1

        plot(X, NM, block=False)
        plot(X, k_neigh(X, 5)[1])

        return adjacency_matrix

    def get_adaptive_neighbors(self, X, index, intrinsic_dim):
        """Determine neighbors by checking tangent space agreement."""
        nn = NearestNeighbors(n_neighbors=20).fit(X)  # Large initial neighborhood
        distances, indices = nn.kneighbors([X[index]])

        neighbors = []
        pca = PCA(n_components=intrinsic_dim)
        pca.fit(X[indices[0][1:]])  # Fit to nearest neighbors excluding self
        plot(X[indices[0][1:]], block= False)
        # plot(pca.components_)

        for i, j in enumerate(indices[0][1:]):
            projection_error = np.linalg.norm(X[j] - pca.inverse_transform(pca.transform([X[j]])))
            # print(projection_error, distances[0][i+1])
            if projection_error < distances[0][i+1] * 0.05:  # Threshold
                neighbors.append(j)


        return neighbors

    def apply_mds(self, geodesic_distances):
        """Apply MDS to project to lower dimensions."""
        mds = MDS(n_components=self.n_components, dissimilarity='precomputed')
        return mds.fit_transform(geodesic_distances)

# Example Usage
if __name__ == "__main__":
    from sklearn.datasets import make_swiss_roll
    X, _ = make_swiss_roll(n_samples=1000, noise=0.1)

    isomap = AdaptiveIsomap(n_components=2)
    embedding = isomap.fit_transform(X)

    import matplotlib.pyplot as plt
    plt.scatter(embedding[:, 0], embedding[:, 1], c=np.linspace(0, 1, X.shape[0]), cmap='Spectral')
    plt.title("Parameterless Isomap with Adaptive Neighborhoods")
    plt.show()
