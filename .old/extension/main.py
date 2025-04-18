import random

import numpy as np
import sklearn.datasets

import inner
import visualise
import inter
import mvu


random_state = 11
random.seed(random_state)
np.random.seed(random_state)

X = [
    mvu.load_teapots(),
    sklearn.datasets.make_s_curve(400, random_state=random_state)[0] + [5, 0, 0],
    # (sklearn.datasets.make_swiss_roll(400, random_state=random_state)[0] * 0.1) + np.array([5, 5, 5]),
    np.array([[
        random.normalvariate(0, 1),
        random.normalvariate(5, 0.5),
        0
        ] for _ in range(100)])
]



X_dim_expanded = X.copy()
d = max([len(X_i[0]) for X_i in X_dim_expanded])
for i in range(len(X_dim_expanded)):
    # print(X_dim_expanded[i].shape, np.zeros((X_dim_expanded[i].shape[0], d - X_dim_expanded[i].shape[1])).shape)
    while X_dim_expanded[i].shape[1] != d:
        X_dim_expanded[i] = np.hstack([X_dim_expanded[i], np.zeros((X_dim_expanded[i].shape[0], d - X_dim_expanded[i].shape[1]))])

X = X_dim_expanded

C = inter.inter_connections(X)


S = []
for X_i in X:
    eps = 1e-3
    n_neighbors = 3

    NG = mvu.kng(X_i - X_i.mean(0), n_neighbors)
    mvu.check_connected(NG)

    K_i = mvu.cached_mvu(X_i - X_i.mean(0), NG, eps)
    S_i = mvu.embed(K_i)
    S.append(S_i)
# S = [mvu.embed(mvu.cached_mvu(X_i - X_i.mean(0), mvu.kng(X_i - X_i.mean(0), n_neighbors=3), eps= 1e-2)) for X_i in X]

print([S_i.shape for S_i in S])
d = max([len(S_i[0]) for S_i in S])
print(d)

print(C)



# print([S_i.shape for S_i in S])

# S_stacked = np.vstack([S_i for S_i in S])
# print(S_stacked.shape)
# visualise.plot(S_stacked)

# X_stacked = np.vstack([X_i for X_i in X])

R = inner.inner_connections(S, C)

print(C, R)

selected_idx = []
for i in range(len(S)):
    for j in range(len(C[i])):
        if i != j and int(C[i][j]) not in selected_idx:
            # print(sum([len(X[l]) for l in range(len(X)) if l < i]), C[i][j])
            selected_idx.append(int(sum([len(X[k]) for k in range(len(X)) if k < i]) + C[i][j]))
            # selected_idx.append(int(np.where(np.all(X_stacked == X[i][int(C[i][j])], axis=1))[0]))
    for k in range(len(R[i])):
        if int(R[i][k]) not in selected_idx:
            # print(sum([len(X[l]) for l in range(len(X)) if l < i]), R[i][k])
            selected_idx.append(int(sum([len(X[l]) for l in range(len(X)) if l < i]) + R[i][k]))
            # selected_idx.append(int(np.where(np.all(X_stacked == X[i][int(R[i][k])], axis=1))[0]))

selected_idx.sort()

print(selected_idx)
# breakpoint()

mvu.plot(np.vstack([X_i for X_i in X]), np.zeros((2,2)))
# visualise.plot(X_stacked, X_stacked[selected_idx])




