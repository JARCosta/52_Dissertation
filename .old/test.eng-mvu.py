import numpy as np
import networkx as nx
from scipy.spatial import distance_matrix
from scipy.linalg import svd
import cvxpy as cvx
from cvxpy import Variable, Maximize, Problem

def build_knn_graph(X, k=5):
    """ Construct k-NN graph using Euclidean distances """
    dist_matrix = distance_matrix(X, X)
    np.fill_diagonal(dist_matrix, np.inf)  # Avoid self-connections
    G = nx.Graph()
    
    for i in range(len(X)):
        knn_indices = np.argsort(dist_matrix[i])[:k]
        for j in knn_indices:
            G.add_edge(i, j, weight=dist_matrix[i, j])
    
    return G

def enhanced_neighborhood_graph(G, X, d=2, ξ=0.95):
    """ Apply ENG (Algorithm 1) to connect disconnected components """
    components = list(nx.connected_components(G))
    
    while len(components) > 1:
        
        # Find nearest disconnected components
        min_dist, best_pair = np.inf, None
        for i, comp_p in enumerate(components):
            for j, comp_q in enumerate(components):
                if i >= j: continue
                p, q = min(comp_p), min(comp_q)  # Select first point in each component
                dist = np.linalg.norm(X[p] - X[q])
                if dist < min_dist:
                    min_dist, best_pair = (p, q)
        


        # Compute singular value contribution ratio
        p, q = best_pair
        delta = X[p] - X[q]
        U, S, Vt = svd(delta.reshape(-1, 1), full_matrices=False)
        eta_d_l = np.sum(S[:d]) / np.sum(S)
        
        if eta_d_l >= ξ:
            G.add_edge(p, q, weight=min_dist)  # Connect components
        
        components = list(nx.connected_components(G))
    return G

def mvu(X, G):
    """ Apply Maximum Variance Unfolding (MVU) using Semidefinite Programming """
    n = len(X)
    K = Variable((n, n), PSD=True)  # Kernel matrix

    _X = cvx.Constant((X @ X.T) * 1e-6)

    
    # Preserve distances in the neighborhood graph
    constraints = []
    for i, j in G.edges():
        # constraints.append(K[i, i] + K[j, j] - 2 * K[i, j] == np.linalg.norm(X[i] - X[j])**2)
        constraints.append(
            (_X[i,i] - 2 * _X[i,j] + _X[j,j]) - 
            (K[i, i] - 2 * K[i, j] + K[j, j]) == 0
        )
    # Objective: Maximize variance (trace of K)
    obj = Maximize(cvx.trace(K))
    prob = Problem(obj, constraints)
    prob.solve(solver="SCS", verbose=True)

    # Compute low-dimensional embedding via eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(K.value)
    idx = np.argsort(-eigvals)[:2]  # Take top 2 eigenvectors
    return eigvecs[:, idx] * np.sqrt(eigvals[idx])

# Example usage
X = np.random.rand(200, 3)  # Random 3D dataset
n = X.shape[0]
k = 5
G = build_knn_graph(X, k)
G = enhanced_neighborhood_graph(G, X)
from plot import plot
NM = np.zeros((n,n))
for (i, j) in G.edges():
    NM[i][j] = 1
plot(X, NM)
X_mvu = mvu(X, G)
