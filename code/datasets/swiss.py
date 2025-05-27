import numpy as np

def default(n_points, noise, random_state=None):
    np.random.seed(random_state) if random_state is not None else None

    t = (3 * np.pi / 2) * (1 + 2 * np.random.rand(n_points, 1))
    height = 30 * np.random.rand(n_points, 1)
    X = np.hstack([
        t * np.cos(t),
        height,
        t * np.sin(t)
    ]) + noise * np.random.randn(n_points, 3)

    labels = np.remainder(np.sum(np.hstack([np.round(t / 2), np.round(height / 12)]), axis=1), 2)
    t = np.hstack([t, height])

    return X, labels, t

def broken(n_points, noise, random_state=None):
    np.random.seed(random_state) if random_state is not None else None
    
    # t1 = (3 * np.pi / 2) * (1 + 2 * np.random.rand(int(np.ceil(n_points / 2)), 1) * 0.4)
    # t2 = (3 * np.pi / 2) * (1 + 2 * (np.random.rand(int(np.floor(n_points / 2)), 1) * 0.4 + 0.6))
    t1 = (3 * np.pi / 2) * (1 + 2 * np.random.rand(n_points, 1) * .4)
    t2 = (3 * np.pi / 2) * (1 + 2 * (np.random.rand(n_points, 1) * .4 + .6))
    t = np.vstack([t1, t2])
    height = 30 * np.random.rand(t.shape[0], 1)
    X = np.hstack([
        t * np.cos(t),
        height,
        t * np.sin(t)
    ]) + noise * np.random.randn(t.shape[0], 3)
    
    labels = np.remainder(np.sum(np.hstack([np.round(t / 2), np.round(height / 12)]), axis=1), 2)
    t = np.hstack([t, height])

    return X, labels, t

def parallel(n_points, noise, random_state=None):
    X1, labels1, t1 = default(n_points, noise, random_state)
    X2, labels2, t2 = default(n_points, noise, random_state)
    X2[:, 1] += 60
    
    X = np.vstack([X1, X2])
    labels = np.hstack([labels1, labels2])
    t = np.vstack([t1, t2])
    t = np.vstack([np.ones((X.shape[0], 1)), 2 * np.ones((X.shape[0], 1))])

    return X, labels, t

def two(n_points, noise, random_state=None):
    np.random.seed(random_state) if random_state is not None else None

    X1, labels1, t1 = default(n_points, noise, random_state)
    X2, labels2, t2 = default(n_points, noise, random_state)
    
    angle_radians = - np.pi / 4
    rotation_matrix = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians), 0],
        [np.sin(angle_radians), np.cos(angle_radians), 0],
        [0, 0, 1]
    ])

    X1 = X1 @ rotation_matrix.T
    
    X1[:, 0] += 20
    X1[:, 1] += 20
    X1[:, 2] += 30


    X2[:, 1] += -20

    X = np.vstack([X1, X2])
    labels = np.hstack([labels1, labels2])
    t = np.vstack([t1, t2])
    # t = np.hstack([np.ones((X.shape[0], 1)), 2 * np.ones((X.shape[0], 1))])

    return X, labels, t

def changing(n_points, noise, random_state=None):
    np.random.seed(random_state) if random_state is not None else None

    r = np.zeros(n_points)
    for i in range(n_points):
        while True:
            rr = np.random.rand()
            if np.random.rand() > rr:
                r[i] = rr
                break
    t = (3 * np.pi / 2) * (1 + 2 * r[:, None])
    height = 21 * np.random.rand(n_points, 1)
    X = np.hstack([t * np.cos(t), height, t * np.sin(t)]) + noise * np.random.randn(n_points, 3)
    labels = np.remainder(np.sum(np.hstack([np.round(t / 2), np.round(height / 10)]), axis=1), 2)

    return X, labels, t

def toro(n, noise, random_state=None):
    np.random.seed(random_state) if random_state is not None else None

    divisions = 3
    division_width = 0.25
    blank_space_width = (1 - (divisions * division_width)) / divisions

    theta = np.vstack(
        [(2*np.pi) * (division_width*np.random.rand(n%divisions, 1))] +
        [(2*np.pi) * (i*blank_space_width + i*division_width + division_width*np.random.rand(n//divisions, 1)) for i in range(divisions)]
    )
    theta = np.random.permutation(theta.flatten()).reshape(-1, 1) # randomize the order of the points
    phi = (3 * np.pi / 2) * (1 + 2 * np.random.rand(n, 1))
    
    X = np.hstack([
        (40 + (phi) * (np.cos(phi))) * np.cos(theta),
        (40 + (phi) * (np.cos(phi))) * np.sin(theta),
        (phi) * np.sin(phi)
    ]) + noise * np.random.randn(n, 3)
    labels = np.remainder(np.sum(np.hstack([np.round(4 * theta / np.pi), np.round(2 * phi / np.pi)]), axis=1), 2)
    t = np.hstack([np.sin(theta/2), phi])

    return X, labels, t
