import numpy as np


def helix(n, noise):

    t = np.linspace(1, n, n)[:, None] / n
    t = (t ** 1.0) * 2 * np.pi
    X = np.hstack([(2 + np.cos(8 * t)) * np.cos(t), (2 + np.cos(8 * t)) * np.sin(t), np.sin(8 * t)]) + noise * np.random.randn(n, 3)
    labels = np.remainder(np.round(t * 1.5), 2)

    return X, labels, t

def twinpeaks(n, noise):

    inc = 1.5 / np.sqrt(n)
    # xx, yy = np.meshgrid(np.arange(-1, 1 + inc, inc), np.arange(-1, 1 + inc, inc))
    xy = 1 - 2 * np.random.rand(2, n)
    X = np.hstack([xy.T, (np.sin(np.pi * xy[0, :]) * np.tanh(3 * xy[1, :]))[:, None]]) + noise * np.random.randn(n, 3)
    X[:, 2] *= 10
    labels = np.remainder(np.sum(np.round((X - np.min(X, axis=0)) / 10), axis=1), 2)

    return X, labels, None

def difficult(n, noise):

    no_dims = 5
    no_points_per_dim = round(n ** (1 / no_dims))
    l = np.linspace(0, 1, no_points_per_dim)
    t = np.array(np.meshgrid(*[l] * no_dims)).T.reshape(-1, no_dims)
    X = np.hstack([np.cos(t[:, 0])[:, None], np.tanh(3 * t[:, 1])[:, None], (t[:, 0] + t[:, 2])[:, None], 
                    (t[:, 3] * np.sin(t[:, 1]))[:, None], (np.sin(t[:, 0] + t[:, 4]))[:, None], 
                    (t[:, 4] * np.cos(t[:, 1]))[:, None], (t[:, 4] + t[:, 3])[:, None], 
                    t[:, 1][:, None], (t[:, 2] * t[:, 3])[:, None], t[:, 0][:, None]])
    X += noise * np.random.randn(*X.shape)
    labels = np.remainder(np.sum(1 + np.round(t), axis=1), 2)

    return X, labels, None

def clusters_3d(n, noise):

    num_clusters = 5
    centers = 10 * np.random.rand(num_clusters, 3)
    D = np.linalg.norm(centers[:, None] - centers[None, :], axis=2)
    min_distance = np.min(D[D > 0])
    X = np.zeros((n, 3))
    labels = np.zeros(n, dtype=int)
    k = 0
    n2 = n - (num_clusters - 1) * 9
    for i in range(num_clusters):
        for _ in range(int(np.ceil(n2 / num_clusters))):
            if k < n:
                X[k] = centers[i] + (np.random.rand(3) - 0.5) * min_distance / np.sqrt(12)
                labels[k] = i + 1
                k += 1
    X += noise * np.random.randn(n, 3)

    return X, labels, None

def intersect(n, noise):

    t = np.linspace(1, n, n)[:, None] / n * (2 * np.pi)
    x = np.cos(t)
    y = np.sin(t)
    height = np.random.rand(len(x), 1) * 5
    X = np.hstack([x, x * y, height]) + noise * np.random.randn(n, 3)
    labels = np.remainder(np.sum(np.hstack([np.round(t / 2), np.round(height / 2)]), axis=1), 2)

    return X, labels, None