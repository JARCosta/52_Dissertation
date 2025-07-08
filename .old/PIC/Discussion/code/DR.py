from matplotlib import pyplot as plt
import numpy as np
from sklearn.manifold import Isomap
import random
# Generate a linear 3D dataset
np.random.seed(11)
random.seed(11)
n_points = 100
x = np.random.uniform(0, 10, n_points)
y = 2 * x + [random.normalvariate(0, 2) for _ in range(n_points)]  # Adding a bit of noise
z = 3 * x - [random.normalvariate(0, 2) for _ in range(n_points)]  # Adding a bit of noise
X = np.vstack((x, y, z)).T  # Combine into a 3D dataset
t = x  # Use x as the color parameter for consistency

# Define the 3D plotting function with consistent coloring
def plot_3D(component_1, component_2, component_3, color, block=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.tick_params(color="white", labelcolor="white")
    scatter = ax.scatter(component_1, component_2, component_3, c=color, cmap='Spectral', alpha=0.6)
    plt.show(block=block)

# Plot the original 3D data
plot_3D(X[:, 0], X[:, 1], X[:, 2], t, False)

# Apply Isomap to reduce dimensionality to 2D
isomap = Isomap(n_neighbors=10, n_components=2)
X_flattened = isomap.fit_transform(X)

# Define the 2D plotting function with consistent coloring
def plot_2D(component_1, component_2, color):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(component_1, component_2, c=color, cmap='Spectral', alpha=0.6)
    plt.xticks([])
    plt.yticks([])
    plt.show()

# Plot the flattened 2D data
plot_2D(X_flattened[:, 0], X_flattened[:, 1], t)
