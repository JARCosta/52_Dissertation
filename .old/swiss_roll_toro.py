import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def circular_swiss_roll(n, noise=0.0):
    divisions = 3
    division_width = 0.25
    blank_space_width = (1 - (divisions * division_width)) / divisions

    theta = np.vstack(
        [(2*np.pi) * (i*blank_space_width + i*division_width + division_width*np.random.rand(n//divisions, 1)) for i in range(divisions)] + 
        [(2*np.pi) * (division_width*np.random.rand(n%divisions, 1))]
    )
    phi = (3 * np.pi / 2) * (1 + 2 * np.random.rand(n, 1))
    
    X = np.hstack([
        (40 + (phi) * (np.cos(phi))) * np.cos(theta),
        (40 + (phi) * (np.cos(phi))) * np.sin(theta),
        (phi) * np.sin(phi)
    ]) + noise * np.random.randn(n, 3)
    labels = np.remainder(np.sum(np.hstack([np.round(4 * theta / np.pi), np.round(2 * phi / np.pi)]), axis=1), 2)
    t = np.hstack([np.sin(theta/2), phi])

    return X, labels, t

if __name__ == '__main__':
    n_samples = 1000
    main_circle_radius = 100
    roll_r = 0.5
    h_scale = 10
    noise_level = 0

    X, labels, t_orig = circular_swiss_roll(n_samples, noise=noise_level)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap='viridis', s=20)
    scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=t_orig.T[1].T, cmap='viridis', s=20)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Closed Circular Swiss Roll')
    fig.colorbar(scatter, label='Labels')
    plt.axis('equal')
    plt.show()