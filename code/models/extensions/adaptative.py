import numpy as np
from sklearn.neighbors import NearestNeighbors

class Adaptative:
    def __init__(self, k_max, eta):
        self.k_min = self.n_neighbors
        self.k_max = k_max
        self.eta = eta
        self.d = self.n_components

    def _neigh_matrix(self, X):

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
                    U, S, V = np.linalg.svd(X_i_k_centered)
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
                U, S, Q_i = np.linalg.svd(X_i_centered)
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


        adjacency_matrix, max_r_i_k = neighborhood_contraction_matrix(X, self.k_max, self.k_min, self.d)
        # print([np.count_nonzero(neigh) for neigh in adjacency_matrix])
        eta = compute_eta_matrix(X, adjacency_matrix, self.d)
        # print(f"Computed Eta: {eta}, Max r_i_k: {max_r_i_k}")
        adaptive_adjacency_matrix = neighborhood_expansion_matrix(X, adjacency_matrix, self.k_max, eta, self.d)
        # print([np.count_nonzero(neigh) for neigh in adaptive_adjacency_matrix])
        # print(adaptive_adjacency_matrix)
        return adaptive_adjacency_matrix


class Adaptative2:
    def __init__(self, k_max, eta):
        self.k_min = self.n_neighbors
        self.k_max = k_max
        self.eta = eta
        self.d = self.n_components

    def _neigh_matrix(self, X):
        """
        Computes the neighborhood for each point using neighborhood contraction.

        Args:
            X: Data matrix, shape (n_samples, n_features).
            k_max: Initial maximum number of neighbors.
            k_min: Minimum allowed number of neighbors.
            eta: Threshold for the ratio of singular values.
            d: Intrinsic dimensionality of the manifold.

        Returns:
            A list of lists, where each inner list contains the indices of the neighbors
            for the corresponding data point.
        """
        k_max = self.k_max
        k_min = self.k_min
        d = self.d

        def compute_eta(X):
            """
            Computes a reasonable value for eta (η) based on the method described in the paper.

            Args:
                X: Data matrix, shape (n_samples, n_features).
                k_values: A list of k values for k-NN neighborhoods (e.g., [5, 10, 15, 20]).
                d: Intrinsic dimensionality of the manifold.

            Returns:
                A reasonable value for eta (η).
            """
            k_values = range(self.k_min, self.k_min + 1)  # Range of k values for k-NN neighborhoods
            d = self.d

            r_values = []

            for k in k_values:
                for i in range(X.shape[0]):
                    # 1. Determine k-NN neighborhood
                    distances = np.linalg.norm(X - X[i], axis=1)
                    neighbors_idx = np.argsort(distances)[:k]
                    X_k = X[neighbors_idx]

                    # 2. Compute singular values and ratio (ρ)
                    X_centered = X_k - np.mean(X_k, axis=0)
                    U, S, V = np.linalg.svd(X_centered)
                    singular_values = S

                    numerator = np.sqrt(np.sum(singular_values[d:]**2))
                    denominator = np.sqrt(np.sum(singular_values[:d]**2))
                    r = numerator / denominator if denominator != 0 else np.inf
                    r_values.append(r)

            # 3. Sort the ρ values
            r_values.sort(reverse=True)

            # 4. Find the largest gap
            max_gap = 0
            eta_index = 0
            for i in range(len(r_values) - 1):
                gap = r_values[i] - r_values[i+1]
                if gap > max_gap:
                    max_gap = gap
                    eta_index = i

            # 5. Method 1: Use the ρ value before the largest gap
            eta_1 = r_values[eta_index] if r_values else 0  # Handle empty list case

            # 6. Method 2: Average the ρ values with the largest gap
            eta_2 = (r_values[eta_index] + r_values[eta_index+1]) / 2 if len(r_values) > 1 else 0 # Handle short list

            # You can choose either eta_1 or eta_2, or return both
            return eta_1, eta_2  # Returning the average for this example

        print(f"Computed eta: {compute_eta(X)}")
        eta = compute_eta(X)[1]

        N = X.shape[0]  # Number of data points
        neigh_matrix = np.zeros((N, N), dtype=int)  # Initialize neighborhood matrix

        for i in range(N):
            # 1. Determine initial k-NN neighborhood
            distances = np.linalg.norm(X - X[i], axis=1)  # Calculate distances to all points
            distances[i] = np.inf
            neighbors_idx = np.argsort(distances)[:k_max]  # Get indices of k_max nearest neighbors
            X_k = X[neighbors_idx]  # Get the neighborhood points
            k = k_max

            while k >= k_min:
                # 2. Compute singular values and ratio
                X_centered = X_k - np.mean(X_k, axis=0)  # Center the neighborhood
                U, S, V = np.linalg.svd(X_centered)
                sigma_i = S

                # print(sigma_i[d:], sigma_i[:d])
                numerator = np.sum(sigma_i[d:]**2)
                denominator = np.sum(sigma_i[:d]**2)
                r = np.sqrt(numerator / denominator)

                # 3. Check the condition
                if r < eta:
                    neigh_matrix[i, neighbors_idx[:k]] = 1  # Update neighborhood matrix
                    # print(f"Neighborhood found for point {i}: {neighbors_idx[:k].tolist()}")
                    break  # Exit the while loop, neighborhood found

                # 4. Contract neighborhood or go to step 5 if k_min is reached
                if k > k_min:
                    k -= 1
                    neighbors_idx = neighbors_idx[:k]
                    X_k = X[neighbors_idx]
                else:
                    # 5. Compute argmin of r (simplified: just take the smallest neighborhood)
                    neigh_matrix[i, neighbors_idx[:k]] = 1  # Update neighborhood matrix
                    # print(f"Neighborhood found for point {i} (min k): {neighbors_idx[:k].tolist()}")
                    break  # Exit the while loop
        
        print(f"Contracted neighborhood matrix: {np.count_nonzero(neigh_matrix)}/{X.shape[0]*k_max} connections")

        """
        Expands the neighborhood based on the algorithm in the paper, using a neighborhood matrix.

        Args:
            X: Data matrix, shape (n_samples, n_features).
            neigh_matrix: Binary neighborhood matrix from the contraction step.
            eta: Threshold for the ratio of singular values.

        Returns:
            A binary neighborhood matrix representing the expanded neighborhood.
        """
        N = X.shape[0]
        expanded_neigh_matrix = neigh_matrix.copy()  # Start with the contracted neighborhoods

        for i in range(N):
            neighborhood_idx = np.where(neigh_matrix[i] == 1)[0] # Indices of the contracted neighborhood
            X_neighborhood = X[neighborhood_idx]
            x_mean = np.mean(X_neighborhood, axis=0)
            Q, _ , _ = np.linalg.svd(X_neighborhood - x_mean, full_matrices=False)
            # print(Q)
            # Q, _ , _ = np.linalg.svd(X_neighborhood - x_mean)
            # print(Q)
            Q = Q[:neighborhood_idx.shape[0]]

            # breakpoint()

            print(f"{i}/{N}", end="\r")
            for j in range(N):
                if neigh_matrix[i, j] == 0:  # If point j is not in the contracted neighborhood
                    x_j = X[j]
                    print(Q.T, (x_j - x_mean))
                    breakpoint()
                    theta_j = Q.T @ (x_j - x_mean)

                    # Check the condition for expansion (Equation 3.11)
                    print(np.linalg.norm(x_j - x_mean - Q @ theta_j), eta * np.linalg.norm(theta_j))
                    if np.linalg.norm(x_j - x_mean - Q @ theta_j) <= eta * np.linalg.norm(theta_j):
                        expanded_neigh_matrix[i, j] = 1  # Add the point to the neighborhood


        print(f"Expanded neighborhood matrix: {np.count_nonzero(expanded_neigh_matrix)} connections")
        breakpoint()
        return expanded_neigh_matrix
