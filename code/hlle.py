import numpy as np
from scipy.linalg import eigh, svd

def hessian_lle(X, n_neighbors=12, n_components=2):
    n_samples, n_features = X.shape
    neighbors = np.zeros((n_samples, n_neighbors), dtype=int)

    # Step 1: Find nearest neighbors
    for i in range(n_samples):
        distances = np.sum((X - X[i])**2, axis=1)
        neighbors[i] = np.argsort(distances)[1:n_neighbors+1]

    # Step 2: Local Tangent Coordinates and Hessian Estimator
    H_rows = []
    for i in range(n_samples):
        Xi = X[neighbors[i]]
        Xi_centered = Xi - np.mean(Xi, axis=0)
        U, S, Vt = svd(Xi_centered, full_matrices=False)
        local_coords = Xi_centered @ Vt[:n_components].T  # local tangent space

        # Construct the Hessian estimator matrix
        G = []
        for coord in local_coords:
            g_row = [1.0]  # constant term
            g_row.extend(coord)  # linear terms
            g_row.extend(coord[i]*coord[j] for i in range(n_components) for j in range(i, n_components))  # quadratic terms
            G.append(g_row)

        G = np.array(G)
        # Compute nullspace of G (right singular vector with smallest singular value)
        _, _, Vg = svd(G)
        null_vecs = Vg[-(n_components * (n_components + 1)) // 2:]  # The Hessian estimator space
        H_row = np.zeros(n_samples)
        H_row[neighbors[i]] = null_vecs[0]  # weights from null space
        H_rows.append(H_row)

    # Step 3: Assemble the Hessian estimator matrix
    H = np.vstack(H_rows)
    # Compute the matrix HH^T
    HHT = H @ H.T

    # Step 4: Eigen decomposition
    evals, evecs = eigh(HHT)
    Y = evecs[:, 1:n_components+1]  # skip the smallest eigenvalue/vector

    return Y


# Generate synthetic data (e.g., swiss roll)
from sklearn.datasets import make_swiss_roll
import matplotlib.pyplot as plt

X, _ = make_swiss_roll(1000, noise=0.05)

Y = hessian_lle(X, n_neighbors=20, n_components=2)

plt.scatter(Y[:, 0], Y[:, 1], c=_, cmap=plt.cm.Spectral)
plt.title("Hessian LLE Embedding")
plt.show()
