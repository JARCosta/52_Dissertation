from matplotlib import pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
from sklearn.manifold import Isomap



# Generate the S-curve dataset
X, t = sklearn.datasets.make_s_curve(1000, random_state=11)



X = X - X.mean(axis=0)


# x_new = np.array([0,0,1])
x_new = np.array([0,0,1.5])


D = (X - x_new) @ (X - x_new).T
d = np.argmin(np.diag(D))

print(d, X[d])

t_new = t[d]

print(x_new, t_new)
print(X.shape)
X = np.vstack([X, x_new])
print(X.shape)
print(t.shape)
t = np.hstack([t, t_new])
print(t.shape)


# Define the 3D plotting function with consistent coloring
def plot_3D(component_1, component_2, component_3, color, block=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(component_1, component_2, component_3, c=color, cmap='Spectral', alpha=0.6)
    # plt.colorbar(scatter, label='Parameter t')
    ax.plot([x_new[0], X[d][0]], [x_new[1], X[d][1]], [x_new[2], X[d][2]], color="red")
    # ax.set_title("3D S-Curve Data")
    ax.tick_params(color="white", labelcolor="white")
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

    print(X_flattened[-1], X_flattened[d])
    
    print([X_flattened[-1][0], X_flattened[d][0]], [X_flattened[-1][1], X_flattened[d][1]])

    plt.plot([X_flattened[-1][0], X_flattened[d][0]], [X_flattened[-1][1], X_flattened[d][1]], color="red")


    # plt.colorbar(scatter, label='Parameter t')
    # plt.title("2D Flattened Data using Isomap")
    # plt.xlabel("Component 1")
    # plt.ylabel("Component 2")
    plt.xticks([])
    plt.yticks([])
    plt.show()

# Plot the flattened 2D data
plot_2D(X_flattened[:, 0], X_flattened[:, 1], t)
