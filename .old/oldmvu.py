import numpy as np
import cvxpy as cvx
import networkx as nx
from scipy.linalg import pinv
from scipy.spatial.distance import cdist
from sklearn.neighbors import kneighbors_graph


from plot import plot
from utils import k_neigh, embed, fast_embed

from models.neighbourhood import Neighbourhood


class MVU:
    def __init__(self, n_neighbors:int=5, eps:float=1e-3):
        self.n_neighbors = n_neighbors
        self.eps = eps
        self.embedding_ = None
        self._mode = 0
    
    def fit(self, X:np.ndarray, starting_K:np.ndarray=None):
        """Fit the Isomap model and compute the low-dimensional embeddings."""

        # inner product matrix of the original data
        _X = cvx.Constant((X @ X.T) * 1e-6)
        n = X.shape[0]

        # inner product matrix of the projected points; PSD constrains G to be both PSD and symmetric
        K = cvx.Variable((n, n), PSD=True)
        if type(starting_K) == type(None):
            K.value = np.zeros((n, n))
        else:
            K.value = np.zeros((n, n))
            K.value = starting_K #TODO: why aren't previously computed K's positive semidefinite?
        print("Goal shape:", K.shape)

        # spread out points in target manifold
        objective = cvx.Maximize(cvx.trace(K))
        constraints = [cvx.sum(K) == 0]

        NM = k_neigh(X)[1]

        # add distance-preserving constraints
        for i in range(n):
            for j in range(n):
                if NM[i, j] != 0:
                    if self._mode == 0:
                        constraints.append(
                            (_X[i,i] - 2 * _X[i,j] + _X[j,j]) - 
                            (K[i, i] - 2 * K[i, j] + K[j, j]) == 0
                        )
                    elif self._mode == 1:
                        constraints.append(
                            (_X[i,i] - 2 * _X[i,j] + _X[j,j]) - 
                            (K[i, i] - 2 * K[i, j] + K[j, j]) >= 0
                        )



        print("Length of constraints:",len(constraints))

        plot(X, NM, title="Default MVU input")

        problem = cvx.Problem(objective, constraints)
        problem.solve(solver="SCS", verbose=True, eps=self.eps)
        self.embedding_ = embed(K.value)

        plot(self.embedding_, NM, title="Default MVU output")
        # try:
        #     problem = cvx.Problem(objective, constraints)
        #     for _ in range(10):
        #         problem.solve(solver="SCS", verbose=True, eps=self.eps)
        #         print(problem.status, K.value)
        #         # Y = fast_embed(K.value, 3)
        #         self.embedding_ = embed(K.value)
        #         plot(self.embedding_)
        # except (cvx.SolverError, KeyboardInterrupt) as e:
        #     print(K.value)
        #     print("Keyboard Interrupt")
        
        return self
    
    def fit_transform(self, X):
        """Fit the model and return the computed embeddings."""
        self.fit(X)
        return self.embedding_

class MVUIneq(MVU):
    def __init__(self, n_neighbors = 5, eps = 0.001):
        super().__init__(n_neighbors, eps)
        self._mode = 1

class MVUNystrom:
    def __init__(self, n_neighbors:int, n_subset:int, eps:float=1e-3):
        self.n_neighbors = n_neighbors
        self.eps = eps
        self.n_subset = n_subset

        self.subset_indices = None
        self.X_subset = None
        self.model = MVU(n_neighbors, eps)

        self.embedding_ = None
    
    def fit(self, X:np.ndarray):
        """Fit _ on a subset of points."""
        self.n_samples = X.shape[0]
        self.subset_indices = np.random.choice(self.n_samples, self.n_subset, replace=False)
        self.remaining_indices = np.setdiff1d(np.arange(self.n_samples), self.subset_indices)
        
        X_subset = X[self.subset_indices]
        self.X_subset = X_subset  # Store subset for later use

        self.model.fit(X_subset)
        self.embedding_ = self.model.embedding_

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

        # Y = fast_embed(self.embedding_, 3)
        # NM = k_neigh(X[self.subset_indices], self.n_neighbors)[1]
        
        # plot(X[self.subset_indices], block=False, title="Executing default isomap")
        # plot(self.embedding_, block=False, title="default isomap executed")

        # Get embeddings for the remaining points using Nyström
        X_remaining = X[self.remaining_indices]
        Y_remaining = self.transform(X_remaining)

        # Combine embeddings of subset and remaining points
        Y_full = np.zeros((self.n_samples, self.embedding_.shape[1]))
        Y_full[self.subset_indices] = self.embedding_
        Y_full[self.remaining_indices] = Y_remaining

        colors = np.zeros(X.shape[0])
        colors[self.subset_indices] = 1

        plot(Y_full, block=False, c=colors, title="Final Nystrom")


        return Y_full




def construct_neighborhood_graph(X, k=5):
    """ Construct the k-nearest neighbors graph """
    knn_graph = kneighbors_graph(X, k, mode='connectivity', include_self=False)
    # return nx.from_scipy_sparse_array(knn_graph)
    return nx.Graph(knn_graph)


def get_connected_components(graph):
    """ Get connected components from the graph """
    return [list(comp) for comp in nx.connected_components(graph)]


def find_nearest_component_pair(X, comp1, comp2):
    """ Find the nearest pair of points between two disconnected components """
    dist_matrix = cdist(X[comp1], X[comp2])  # Compute distances
    min_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
    return comp1[min_idx[0]], comp2[min_idx[1]]


def enhanced_neighborhood_graph(X, k=5):
    """ Connect disconnected components iteratively to form a single connected graph """
    graph = construct_neighborhood_graph(X, k)
    components = get_connected_components(graph)

    while len(components) > 1:
        new_edges = []
        for i in range(len(components) - 1):
            comp1, comp2 = components[i], components[i + 1]
            p1, p2 = find_nearest_component_pair(X, comp1, comp2)
            new_edges.append((p1, p2))

        graph.add_edges_from(new_edges)
        components = get_connected_components(graph)

    return graph


class ENGMVU(MVU):
    def __init__(self, n_neighbors=5, eps=1e-3):
        super().__init__(n_neighbors, eps)

    def fit(self, X: np.ndarray, starting_K: np.ndarray = None):
        """Fit the ENG-MVU model and compute low-dimensional embeddings."""
        # Step 1: Construct an enhanced connected neighborhood graph
        print("Applying Enhanced Neighborhood Graph (ENG) correction...")
        enhanced_graph = enhanced_neighborhood_graph(X, self.n_neighbors)

        # Convert the graph to a nearest neighbors matrix
        adjacency_matrix = nx.to_numpy_array(enhanced_graph)

        # Ensure connectivity in k-NN search
        NM = (adjacency_matrix > 0).astype(int)

        print(np.count_nonzero(NM))
        plot(X, NM, title="ENG-MVU Input Graph")
        

        # Step 2: Proceed with standard MVU using the enhanced graph
        print("Applying Maximum Variance Unfolding (MVU)...")
        _X = cvx.Constant((X @ X.T) * 1e-6)
        n = X.shape[0]

        K = cvx.Variable((n, n), PSD=True)
        K.value = np.zeros((n, n)) if starting_K is None else starting_K

        objective = cvx.Maximize(cvx.trace(K))
        constraints = [cvx.sum(K) == 0]

        for i in range(n):
            for j in range(n):
                if NM[i, j] != 0:
                    constraints.append(
                        (_X[i, i] - 2 * _X[i, j] + _X[j, j]) - (K[i, i] - 2 * K[i, j] + K[j, j]) == 0
                    )

        print(f"Number of distance-preserving constraints: {len(constraints)}")


        problem = cvx.Problem(objective, constraints)
        problem.solve(solver="SCS", verbose=True, eps=self.eps)

        self.embedding_ = embed(K.value)

        plot(self.embedding_, NM, title="ENG-MVU Output Embedding")
        return self



import numpy as np
import cvxpy as cvx
import networkx as nx
from scipy.spatial.distance import cdist
from scipy.linalg import svd, pinv
from utils import k_neigh, embed
import matplotlib.pyplot as plt


class MVU_ENG:
    def __init__(self, n_neighbors=5, eps=1e-3, xi=0.95):
        self.n_neighbors = n_neighbors
        self.eps = eps
        self.xi = xi  # Robustness parameter
        self.embedding_ = None

    def fit(self, X):
        """Compute MVU embeddings and apply Enhanced Neighborhood Graph (ENG)."""
        self.X = X
        n = X.shape[0]

        # Step 1: Compute initial MVU embedding
        self.embedding_ = self.mvu_solve(X, k_neigh(X)[1])

        # Step 2: Construct k-NN graph and detect disconnected components
        # G = k_neigh(self.embedding_)[0]
        G = self.build_knn_graph(self.embedding_, self.n_neighbors)
        components = list(nx.connected_components(G))

        # Step 3: Apply ENG connection strategy
        if len(components) > 1:
            self.embedding_ = self.enhanced_graph_connection(G, components)

        return self

    def mvu_solve(self, X, NM):
        n = X.shape[0]
        _X = cvx.Constant((X @ X.T) * 1e-6)

        K = cvx.Variable((n, n), PSD=True)
        objective = cvx.Maximize(cvx.trace(K))
        constraints = [cvx.sum(K) == 0]

        for i in range(n):
            for j in range(n):
                if NM[i, j] != 0:
                    constraints.append(
                        (_X[i, i] - 2 * _X[i, j] + _X[j, j]) - (K[i, i] - 2 * K[i, j] + K[j, j]) == 0
                    )

        problem = cvx.Problem(objective, constraints)
        problem.solve(solver="SCS", verbose=True, eps=self.eps)
        return embed(K.value)

    def build_knn_graph(self, X_embedded, k):
        """Construct k-nearest neighbor graph from embedded points."""
        n = X_embedded.shape[0]
        G = nx.Graph()
        distances = cdist(X_embedded, X_embedded)
        np.fill_diagonal(distances, np.inf)

        for i in range(n):
            neighbors = np.argsort(distances[i])[:k]
            for j in neighbors:
                G.add_edge(i, j, weight=distances[i, j])

        return G

    def enhanced_graph_connection(self, G, components):
        """Connect disconnected components using ENG strategy."""
        component_pairs = self.get_component_pairs(components)
        new_edges = []

        for (p, q) in component_pairs:
            edges = self.find_best_edges_between_components(p, q)
            new_edges.extend(edges)

        G.add_edges_from(new_edges)

        # Step 5: Recompute MVU embedding with new connections
        self.embedding_ = self.mvu_solve(self.X)
        return self.embedding_

    def get_component_pairs(self, components):
        """Generate pairs of disconnected components."""
        pairs = []
        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                pairs.append((list(components[i]), list(components[j])))
        return pairs

    def find_best_edges_between_components(self, comp_p, comp_q):
        """Select best edges between two disconnected components using ENG strategy."""
        Y_p = self.embedding_[comp_p]  # Extract points in component p
        Y_q = self.embedding_[comp_q]  # Extract points in component q
        m = Y_p.shape[1]  # Dimensionality of the embedding space

        # Compute pairwise distances and sort
        D = cdist(Y_p, Y_q)  # Pairwise Euclidean distances
        sorted_indices = np.dstack(np.unravel_index(np.argsort(D.ravel()), D.shape))[0]

        best_edges = []
        eta_d = self.compute_eta_d()  # Compute η^(d) from all data

        for l in range(1, min(len(comp_p), len(comp_q))):
            # Select top-l closest pairs
            p_idx, q_idx = sorted_indices[l]
            delta = Y_p[p_idx] - Y_q[q_idx]  # Difference vector

            # Compute SVD and contribution ratio
            U, S, Vt = svd(delta.reshape(-1, 1), full_matrices=False)

            d = min(m, l)  # Correcting min(m, l)
            eta_d_l = np.sum(S[:d]) / np.sum(S[:min(m, l)])  # Correct computation

            # Stopping criterion
            if eta_d_l < self.xi * eta_d:
                break  # Stop adding edges

            best_edges.append((comp_p[p_idx], comp_q[q_idx]))

        return best_edges


    def fit_transform(self, X):
        """Fit MVU + ENG and return the embedding."""
        self.fit(X)
        return self.embedding_

