import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_s_curve
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d import Axes3D

# Set random seed for reproducibility
random.seed(11)
np.random.seed(11)

# Generate S-curve data
num_points = 200
X, t = make_s_curve(num_points, noise=0.1)  # Generate 3D S-curve data

# Use k-Nearest Neighbors to define connections
k = 5  # Number of nearest neighbors
nbrs = NearestNeighbors(n_neighbors=k).fit(X)
distances, indices = nbrs.kneighbors(X)

# Create a connection matrix based on kNN
connection_matrix = np.zeros((num_points, num_points), dtype=int)
for i in range(num_points):
    for j in indices[i]:
        connection_matrix[i, j] = 1

# Select a point to highlight
highlight_index = 0
highlight_point = X[highlight_index]

# Plot the points and connections in 3D
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Draw connections
for i in range(num_points):
    for j in range(num_points):
        if connection_matrix[i, j] == 1:
            if i == highlight_index or j == highlight_index:
                ax.plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]], [X[i, 2], X[j, 2]],
                        color="red", linewidth=2)
            else:
                ax.plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]], [X[i, 2], X[j, 2]],
                        color="blue", alpha=0.2)

# Draw all points
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=t, cmap="Spectral", s=50, label="Points")

# Highlight the selected point
ax.scatter(highlight_point[0], highlight_point[1], highlight_point[2],
           color="red", s=100, label="Highlighted Point")

# Customize plot
# ax.set_title("3D S-Curve with k-NN Connections")
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
ax.tick_params(color="white", labelcolor="white")
# ax.legend(loc="upper left")
plt.show()
