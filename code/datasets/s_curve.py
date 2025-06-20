import numpy as np

def default(n_points, noise):

    t = (3 * np.pi) * (np.random.rand(n_points, 1)) - 3/2 * np.pi

    # t = 3 * np.pi * (np.random.rand(n_points, 1) - 0.5)
    height = 2 * np.random.rand(n_points, 1)

    X = np.hstack([
        np.sin(t),
        height,
        np.sign(t) * (np.cos(t) - 1)
    ]) + noise * np.random.randn(n_points, 3)

    labels = np.remainder(np.sum(np.hstack([np.round(t / 2), np.round(height / 2)]), axis=1), 2)  # TODO: ajust
    t = np.hstack([t, height])

    return X, labels, t


def broken(n_points, noise, class_splits=2):

    t0 = (0.4 * np.pi) * (np.random.rand(n_points% 4, 1)) - 3/2 * np.pi
    t1 = (0.4 * np.pi) * (np.random.rand(n_points//4, 1)) - 3/2 * np.pi
    t2 = (0.8 * np.pi) * (np.random.rand(n_points//4, 1)) - 3/2 * np.pi + 0.6 * np.pi
    t3 = (0.8 * np.pi) * (np.random.rand(n_points//4, 1)) - 3/2 * np.pi + 1.6 * np.pi
    t4 = (0.4 * np.pi) * (np.random.rand(n_points//4, 1)) - 3/2 * np.pi + 2.6 * np.pi
    t = np.vstack([t0, t1, t2, t3, t4])
    
    height = 2 * np.random.rand(n_points, 1)
    X = np.hstack([
        np.sin(t),
        height,
        np.sign(t) * (np.cos(t) - 1)
    ]) + noise * np.random.randn(n_points, 3)
    
    labels = np.remainder(np.sum(np.hstack([np.round(t / class_splits), np.round(height / class_splits)]), axis=1), 2)
    t = np.hstack([t, height])

    return X, labels, t