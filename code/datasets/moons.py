import numpy as np


def default(n_points, noise):
    t1 = np.pi * np.random.rand(n_points // 2, 1)
    t2 = np.pi * np.random.rand(n_points // 2 + n_points % 2, 1)

    X1 = np.hstack([
        np.sin(t1),
        np.cos(t1)
    ]) + noise * np.random.randn(n_points // 2, 2)
    X2 = np.hstack([
        (1 - np.sin(t2))/4,
        (- np.cos(t2))/4 
    ]) + noise * np.random.randn(n_points // 2 + n_points % 2, 2)

    X = np.vstack([X1, X2])
    labels = np.hstack([np.zeros(n_points // 2), np.ones(n_points // 2 + n_points % 2)])
    labels = labels[:, None] # make labels from (n,) into (n, 1)
    t = np.hstack([t1, t2])

    return X, labels, t


def four(n_points, noise):
    X1, labels1, t1 = default(n_points, noise)
    X2, labels2, t2 = default(n_points, noise)
    X1 = np.hstack([noise * np.random.randn(X1.shape[0], 1), X1])
    X2 = np.hstack([noise * np.random.randn(X1.shape[0], 1), X2])
    X2[:, 0] += 1



    X = np.vstack([X1, X2])
    labels = np.vstack([labels1, labels2])
    t = np.hstack([t1, t2])

    return X, labels, t

