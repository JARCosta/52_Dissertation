import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse.linalg import eigsh
from scipy.sparse import csgraph
from scipy.sparse import csr_matrix

class LaplacianEigenmaps:
    def __init__(self, no_dims=2, k=12, sigma=1.0):
        self.no_dims = no_dims
        self.k = k
        self.sigma = sigma

    def _construct_neighborhood_graph(self, X):
        """Constructs the k-nearest neighbor graph with Euclidean distances."""
        print("Constructing neighborhood graph...")
        G = cdist(X, X, metric='euclidean')  # Compute pairwise distances
        n_samples = G.shape[0]

        # Keep only k-nearest neighbors
        sorted_indices = np.argsort(G, axis=1)

        NM = sorted_indices[:, 1:self.k+1]
        print(NM[0])

        self._W = []
        for i in range(self.n):
            w_aux = np.zeros((1, self.n))


            similarities = []
            for v in NM[i]:
                similarity = np.exp(- (G[i,v]*G[i,v])/self.sigma)
                similarities.append(similarity)

            # print(similarities)
            # breakpoint()
            np.put(w_aux, nn_matrix[i], similarities)
            self._W.append(w_aux[0])
        
        self._W = np.array(self._W)
        

        # print(sorted_indices[0, self.k+1:], len(sorted_indices[0, self.k+1:]))
        # breakpoint()
        
        for i in range(n_samples):
            G[i, sorted_indices[i, self.k+1:]] = 0  # Keep k neighbors
        # print(G[0], np.count_nonzero(G[0]))
        # breakpoint()
        G = np.maximum(G, G.T)  # Ensure symmetry

        G = G ** 2  # Square distances
        self.max_dist = np.max(G)  # Store max distance for later scaling
        G /= self.max_dist  # Normalize distances

        print(G[0], np.count_nonzero(G[0]))

        return csr_matrix(G)

    def _largest_connected_component(self, G):
        """Extracts the largest connected component from the graph."""
        print("Finding largest connected component...")
        n_components, labels = csgraph.connected_components(G, directed=False)
        component_sizes = np.bincount(labels)
        largest_component = np.argmax(component_sizes)
        conn_comp = np.where(labels == largest_component)[0]

        return G[conn_comp, :][:, conn_comp], conn_comp

    def _compute_laplacian(self, G):
        """Computes the graph Laplacian matrix."""
        print("Computing weight matrices...")


        G.data = np.exp(-G.data / (2 * self.sigma ** 2))  # Apply Gaussian kernel

        print(G[0])
        breakpoint()

        D = np.array(G.sum(axis=1)).flatten()  # Degree matrix (diagonal elements)
        D = np.diag(D)
        L = D - G.toarray()  # Compute unnormalized Laplacian
             
        return L, D

    def fit_transform(self, X):
        """Computes the Laplacian Eigenmaps embedding."""
        G = self._construct_neighborhood_graph(X)
        G, conn_comp = self._largest_connected_component(G)
        L, D = self._compute_laplacian(G)

        print("Constructing Eigenmaps...")
        # Solve generalized eigenproblem L v = λ D v
        eigvals, eigvecs = eigsh(L, k=self.no_dims + 1, M=D, which='SM')

        # Sort eigenvalues and eigenvectors
        sorted_indices = np.argsort(eigvals)
        eigvals = eigvals[sorted_indices][1:self.no_dims + 1]  # Ignore first eigenvector
        eigvecs = eigvecs[:, sorted_indices[1:self.no_dims + 1]]

        # Store mapping for out-of-sample extension
        self.mapping = {
            "K": G,
            "vec": eigvecs,
            "val": eigvals,
            "X": X[conn_comp],
            "sigma": self.sigma,
            "k": self.k,
            "conn_comp": conn_comp,
        }

        return eigvecs

# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_swiss_roll
    import matplotlib.pyplot as plt

    # Generate Swiss Roll dataset
    X, color = make_swiss_roll(n_samples=2000, noise=0.1)
    # X = X[:, [0, 2]]  # Reduce to 2D for simplicity

    # Apply Laplacian Eigenmaps
    lap_eig = LaplacianEigenmaps(no_dims=2, k=10, sigma=1.0)
    X_embedded = lap_eig.fit_transform(X)

    # Plot the reduced representation
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=np.arange(len(X_embedded)), cmap='Spectral')
    plt.title("Laplacian Eigenmaps Projection")
    plt.colorbar()
    plt.show()

    from le2 import LE
    # from le import LE
    le = LE(X, dim = 3, k=10, graph = 'k-nearest', weights = 'heat kernel', 
            sigma = 5, laplacian = 'symmetrized')
    # le = LE(X, dim = 3, eps = 1.97, graph = 'eps', weights = 'heat kernel', 
    #         sigma = 5, laplacian = 'symmetrized')
    Y_t = le.transform()

    from plot import plot
    plot(Y_t)

    le.plot_embedding_2d(color, cmap=plt.cm.jet, grid = False, size = (14, 6))


