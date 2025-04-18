
class Nystrom(Isomap):
    def __init__(self, model_args, n_subset:int, n_neighbors:int, n_components:int):
        super().__init__(model_args, n_neighbors, n_components)
        self.n_subset = n_subset
    
    def fit(self, X):
        self.subset_indices = np.random.choice(self.n, self.n_subset, replace=False)
        self.remaining_indices = np.setdiff1d(np.arange(self.n), self.subset_indices)

        X_subset = X[self.subset_indices]
        self.X_subset = X_subset  # Store subset for later use

        NM = self.neigh_matrix(X_subset)
        super().fit(X_subset)

        print(self.embedding_.shape, self.embedding_)

        X_remaining = X[self.remaining_indices]

        # Compute distances from remaining points to subset
        D_remaining_subset = cdist(X_remaining, self.X_subset)

        # Convert distances to a kernel matrix (Gaussian-like transformation)
        sigma = np.std(D_remaining_subset)
        K_remaining_subset = np.exp(-D_remaining_subset**2 / (2 * sigma**2))

        # Compute the pseudo-inverse of the subset kernel
        K_subset = np.exp(-cdist(self.X_subset, self.X_subset)**2 / (2 * sigma**2))
        K_subset_pinv = pinv(K_subset)

        # Compute remaining embeddings
        Y_remaining = K_remaining_subset @ K_subset_pinv @ self.embedding_

        # Combine embeddings of subset and remaining points
        Y_full = np.zeros((self.n_samples, self.embedding_.shape[1]))
        Y_full[self.subset_indices] = self.embedding_
        Y_full[self.remaining_indices] = Y_remaining

        colors = np.zeros(X.shape[0])
        colors[self.subset_indices] = 1

        plot(Y_full, block=False, c=colors, title="Final Nystrom")

        return self



class IsomapNystrom:
    def __init__(self, n_neighbors:int, n_components:int, n_subset:int):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.n_subset = n_subset
        self.isomap = Isomap(n_neighbors, n_components)
        self.subset_indices = None
        self.embedding_ = None
        self.X_subset = None

    def fit(self, X):
        """Fit Isomap on a subset of points."""
        self.n_samples = X.shape[0]
        self.subset_indices = np.random.choice(self.n_samples, self.n_subset, replace=False)
        self.remaining_indices = np.setdiff1d(np.arange(self.n_samples), self.subset_indices)
        
        X_subset = X[self.subset_indices]
        self.X_subset = X_subset  # Store subset for later use

        # plot(X_subset)

        self.isomap.fit(X_subset)
        self.embedding_ = self.isomap.embedding_

        # plot(self.embedding_)

        return self

    def transform(self, X_new):
        """Apply Nyström extension to estimate embeddings for new points."""
        if self.embedding_ is None:
            raise ValueError("Fit the model before calling transform.")

        # Compute distances from new points to subset
        D_new_subset = cdist(X_new, self.X_subset)

        # Convert distances to a kernel matrix (Gaussian-like transformation)
        sigma = np.std(D_new_subset)
        K_new_subset = np.exp(-D_new_subset**2 / (2 * sigma**2))

        # Compute the pseudo-inverse of the subset kernel
        K_subset = np.exp(-cdist(self.X_subset, self.X_subset)**2 / (2 * sigma**2))
        K_subset_pinv = pinv(K_subset)

        # Compute new embeddings
        Y_new = K_new_subset @ K_subset_pinv @ self.embedding_
        # print(self.embedding_.shape)
        # print(Y_new.shape)

        return Y_new

    def fit_transform(self, X):
        """Fit Isomap on a subset and extend to the whole dataset."""
        self.fit(X)

        # Get embeddings for the remaining points using Nyström
        X_remaining = X[self.remaining_indices]
        Y_remaining = self.transform(X_remaining)

        # Combine embeddings of subset and remaining points
        Y_full = np.zeros((self.n_samples, self.n_components))
        Y_full[self.subset_indices] = self.embedding_
        Y_full[self.remaining_indices] = Y_remaining

        colors = np.zeros(X.shape[0])
        colors[self.subset_indices] = 1

        # plot(Y_full, block=False, c=colors, title="Final Nystrom")


        return Y_full

class IsomapENM:
    def __init__(self, n_neighbors:int, n_components:int):
        """
        Enhanced Isomap with improved neighborhood graph construction.

        Parameters:
        - n_neighbors: int, number of neighbors for k-NN graph.
        - n_components: int, number of dimensions for embedding.
        """
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.embedding_ = None

    def _compute_adaptive_k(self, X):
        """Determine adaptive k based on local density (distance to the 10th nearest neighbor)."""
        nbrs = NearestNeighbors(n_neighbors=10).fit(X)
        distances, _ = nbrs.kneighbors(X)
        density = distances[:, -1]  # Distance to the 10th nearest neighbor as a density estimate
        adaptive_k = np.clip((density / np.median(density)) * self.n_neighbors, 2, 2 * self.n_neighbors).astype(int)
        return adaptive_k

    def _compute_geodesic_distance_knn(self, X):
        """Compute geodesic distances using an enhanced k-NN graph (mutual and/or adaptive k-NN)."""
        n = X.shape[0]

        k_vals = self._compute_adaptive_k(X)

        # Construct k-NN graph
        graph = sp.lil_matrix((n, n))  # Sparse graph

        for i in range(n):
            k = k_vals[i]
            nbrs = NearestNeighbors(n_neighbors=k).fit(X)
            distances, indices = nbrs.kneighbors(X[i].reshape(1, -1))

            for j in range(1, k):  # Skip self-loop (j=0)
                # if self.mutual_knn:
                #     # Mutual k-NN: Only add edge if both points are in each other's neighbor list
                #     nbrs_j = NearestNeighbors(n_neighbors=k).fit(X)
                #     _, indices_j = nbrs_j.kneighbors(X[indices[0, j]].reshape(1, -1))

                #     if i in indices_j:
                #         graph[i, indices[0, j]] = distances[0, j]
                #         graph[indices[0, j], i] = distances[0, j]  # Ensure symmetry
                # else:
                    graph[i, indices[0, j]] = distances[0, j]
                    graph[indices[0, j], i] = distances[0, j]  # Ensure symmetry

        # Compute shortest paths using Dijkstra
        D = shortest_path(graph, directed=False, method='D')

        print(np.amax(D))
        if np.amax(D) == np.inf:
            breakpoint()
        return D

    def fit(self, X):
        """Fit the Isomap model and compute the low-dimensional embeddings."""
        D = self._compute_geodesic_distance_knn(X)
        D_sq = D ** 2  # Squared distance matrix
        
        n = D.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n  # Centering matrix
        K = -0.5 * H @ D_sq @ H  # Double centering
    
        self.embedding_ = fast_embed(K, self.n_components, verbose=False)
        # plot(self.embedding_, block=False, title="Result of ENM isomap")
        return self

    def fit_transform(self, X):
        """Fit the model and return the computed embeddings."""
        self.fit(X)
        return self.embedding_


from numpy.linalg import svd

