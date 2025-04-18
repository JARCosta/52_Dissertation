import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from sklearn.neighbors import kneighbors_graph
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform

# Step 1: Generate a Toy Dataset (Two Colors)
np.random.seed(42)
X1 = np.random.randn(10, 3) + np.array([3, 3, 3])  # Class 1 (Red)
X2 = np.random.randn(10, 3) + np.array([-3, -3, -3])  # Class 2 (Blue)
X = np.vstack((X1, X2))
colors = np.array([0]*10 + [1]*10)  # Assign colors (0 = Red, 1 = Blue)
n = X.shape[0]

# Step 2: Compute Pairwise Distance and Nearest Neighbors Graph
D = squareform(pdist(X, 'sqeuclidean'))  # Squared Euclidean distance
W = kneighbors_graph(X, n_neighbors=5, mode='connectivity').toarray()  # k-NN graph

# Step 3: Setup the Semidefinite Programming (SDP) Problem
K = cp.Variable((n, n), symmetric=True)

# MVU Constraint: Preserve distances for neighbors
constraints = [K >> 0]  # Positive semi-definiteness
constraints.append(cp.trace(K) >= 1e-6)  # Avoid trivial solutions
constraints.append(cp.norm(K, 'fro') <= 100)  # Prevents explosion
ones = np.ones((n, n)) / n
constraints.append(K @ ones == 0)  # Ensures centering
NG=np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if W[i, j] > 0:  # If (i, j) are neighbors
            NG[i][j] = 1
            constraints.append(K[i, i] + K[j, j] - 2*K[i, j] == D[i, j])

# Must-Link (Same Color) Constraint: Keep them closer
delta_ML = 1.0  # Small threshold
for i in range(n):
    for j in range(n):
        if colors[i] == colors[j]:  # Same color
            NG[i][j] = 1
            constraints.append(K[i, i] + K[j, j] - 2*K[i, j] <= delta_ML)

# Cannot-Link (Different Colors) Constraint: Keep them apart
# delta_CL = 0.1  # Larger threshold
# for i in range(n):
#     for j in range(n):
#         if colors[i] != colors[j]:  # Different color
            # NG[i][j] = 1
            # constraints.append(K[i, i] + K[j, j] - 2*K[i, j] == delta_CL)

# Objective: Maximize variance (trace of K)
objective = cp.Maximize(cp.trace(K))
problem = cp.Problem(objective, constraints)

from plot import plot
plot(X, NG)

# Step 4: Solve the SDP
problem.solve(solver=cp.SCS, verbose=True)
print(K.value)

# Step 5: Extract the Low-Dimensional Representation
eigenvalues, eigenvectors = np.linalg.eigh(K.value)  # Eigen-decomposition
embedding = eigenvectors[:, -2:]  # Take the top 2 dimensions

# Step 6: Visualize the Results
plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, cmap='coolwarm', edgecolors='k')
plt.title("Colored MVU - 2D Projection")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()
