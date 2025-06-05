import numpy as np
import sklearn.datasets

def swiss(n_points, noise, random_state=None):
    """Sklearn swiss roll dataset with custom labeling."""
    X, t = sklearn.datasets.make_swiss_roll(n_samples=n_points, noise=noise, random_state=random_state)
    t, height = t.reshape((n_points, 1)), X[:, 1].reshape((n_points, 1))

    labels = np.remainder(np.sum(np.hstack([np.round(t / 2), np.round(height / 12)]), axis=1), 2)
    t = np.hstack([t, height])

    return X, labels, t

def s_curve(n_points, noise, random_state=None):
    """Sklearn S-curve dataset with custom labeling."""
    X, t = sklearn.datasets.make_s_curve(n_samples=n_points, noise=noise, random_state=random_state)
    t, height = t.reshape((n_points, 1)), X[:, 1].reshape((n_points, 1))

    labels = np.remainder(np.sum(np.hstack([
        np.round(10 * (t / (np.max(t) - np.min(t)))),
        np.round(2 * (height / (np.max(height) - np.min(height))))
    ]), axis=1), 2)
    labels = np.remainder(np.sum(np.hstack([np.round(t), np.round(height)]), axis=1), 2)
    t = np.hstack([t, height])

    return X, labels, t

def moons(n_points, noise, random_state=None):
    """Sklearn two moons dataset with custom labeling."""
    X, t = sklearn.datasets.make_moons(n_samples=n_points, noise=noise, random_state=random_state)
    t = t.reshape((n_points, 1))
    labels = np.remainder(np.sum(np.round(t), axis=1), 2)

    return X, labels, t 