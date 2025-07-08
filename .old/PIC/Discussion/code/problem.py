from matplotlib import pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
from sklearn.manifold import Isomap

# Generate the Swiss roll dataset
X, t = sklearn.datasets.make_swiss_roll(1000)

# Define a function to make a cut in the Swiss roll
def make_cut(X, axis=0, lower_bound=5, upper_bound=10):
    """
    Removes points from the dataset within a specified range along a given axis.
    axis: The axis along which to make the cut (0: x, 1: y, 2: z).
    lower_bound: The lower bound of the range to remove.
    upper_bound: The upper bound of the range to remove.
    """
    return X[(X[:, axis] < lower_bound) | (X[:, axis] > upper_bound)], t[(X[:, axis] < lower_bound) | (X[:, axis] > upper_bound)]

# Make a cut in the Swiss roll along the z-axis
X_cut, t_cut = make_cut(X, axis=2, lower_bound=5, upper_bound=10)

# Define the 3D plotting function with consistent coloring
def plot_3D(component_1, component_2, component_3, color, block=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.tick_params(color="white", labelcolor="white")
    scatter = ax.scatter(component_1, component_2, component_3, c=color, cmap='Spectral', alpha=0.6)
    plt.show(block=block)

# Plot the original 3D data with the cut applied
plot_3D(X_cut[:, 0], X_cut[:, 1], X_cut[:, 2], t_cut, False)

# Apply Isomap to reduce dimensionality to 2D
isomap = Isomap(n_neighbors=10, n_components=2)
X_flattened = isomap.fit_transform(X_cut)

# Define the 2D plotting function with consistent coloring
def plot_2D(component_1, component_2, color):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(component_1, component_2, c=color, cmap='Spectral', alpha=0.6)
    plt.xticks([])
    plt.yticks([])
    plt.show()

# Plot the flattened 2D data with the cut applied
plot_2D(X_flattened[:, 0], X_flattened[:, 1], t_cut)
# plot_2D(t_cut, X_cut[:, 1], t_cut)
