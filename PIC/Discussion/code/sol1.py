
import random
import threading

from matplotlib import pyplot as plt
import numpy as np
import sklearn
from sklearn.manifold import Isomap
from sklearn.neighbors import NearestNeighbors
import sklearn.datasets

############################################################
############################################################
# FUNCTIONS
############################################################
############################################################

def inter_connections(X:np.ndarray):
    """
    C <- -1_{|X|x|X|}

    for i = 1 to |X| do
        for j = 1 to |X| do
            if i != j do
                r <- 1/|X_j| (X_j.T * 1_d)
                C_{ij} <- argmin diag (X_i - r) @ (X_i - r).T
    """

    C = np.zeros((len(X), len(X))) - 1
    
    for i in range(len(X)):
        for j in range(len(X)):
            if i != j:
                r = X[j].mean(0)
                closest_to_j_mean_coords = X[i][np.argmin(np.diag((X[i] - r) @ (X[i] - r).T))]
                C[i][j] = np.where(np.all(X[i] == closest_to_j_mean_coords, axis=1))[0]

    return C

def mvu_concurrent(data:np.ndarray, NG:np.ndarray, eps=1e-5):
    import cvxpy as cvx

    # inner product matrix of the original data
    X = cvx.Constant((data @ data.T) * 1e-6)
    n = len(data)

    # inner product matrix of the projected points; PSD constrains G to be both PSD and symmetric
    K = cvx.Variable((n, n), PSD=True)
    K.value = np.zeros((n, n))
    print("G shape:", K.shape)

    # spread out points in target manifold
    objective = cvx.Maximize(cvx.trace(K))
    constraints = [cvx.sum(K) == 0]

    # add distance-preserving constraints
    for i in range(n):
        for j in range(n):
            if NG[i, j] != 0:
                constraints.append(
                    (X[i,i] - 2 * X[i,j] + X[j,j]) - 
                    (K[i, i] - 2 * K[i, j] + K[j, j]) == 0
                )

    problem = cvx.Problem(objective, constraints)
    problem.solve(solver="SCS", verbose=True, eps=eps)

    return K.value

def mvu_global(data:np.ndarray, NG:np.ndarray, eps=4e-4):
    import cvxpy as cvx

    # inner product matrix of the original data
    n = len(data)

    # inner product matrix of the projected points; PSD constrains G to be both PSD and symmetric
    K = cvx.Variable((n, n), PSD=True)
    K.value = np.zeros((n, n))
    print("G shape:", K.shape)

    # spread out points in target manifold
    objective = cvx.Maximize(cvx.trace(K))
    constraints = [cvx.sum(K) == 0]

    # add distance-preserving constraints
    for i in range(n):
        for j in range(n):
            if NG[i, j] != 0:
                constraints.append(
                    (NG[i,j]) - 
                    (K[i, i] - 2 * K[i, j] + K[j, j]) == 0
                )

    problem = cvx.Problem(objective, constraints)
    problem.solve(solver="SCS", verbose=True, eps=eps)

    return K.value

def inner_connections(S:np.ndarray, C:np.ndarray):
    """
    R <- {}
    
    for i=0 to |S| do
        A <- 0_|S_i|

        for j=0 to |C_i| do
            if C_{ij} != -1 do
                A <- A + log diag (S_i - C_ij)(S_i - C_ij).T

        for j=|C_i| to d+1 do
            r <- S_{i argmax(A)}
            R_i <- R_i U {r}
            A <- A + log diag (S_i - r) (S_i - r).T
    """

    R = [[] for _ in range(len(S))]

    for i in range(len(S)):
        A = np.zeros(len(S[i]))

        # Consider the already used inter-connection points
        for j in range(len(C[i])):
            if C[i][j] != -1:
                with np.errstate(divide='ignore'):
                    idx = int(C[i][j])
                    A += np.log(np.diag((S[i] - S[i][idx]) @ (S[i] - S[i][idx]).T))


        # Get the rest reference points from inner-connections
        for j in range(len(C[i])-1, len(S[i][0])+1):
            r_idx = np.argmax(A)
            R[i].append(r_idx)
            with np.errstate(divide='ignore'):
                A += np.log(np.diag((S[i] - S[i][r_idx]) @ (S[i] - S[i][r_idx]).T))
    return R


def plot_2D(component_1, component_2, color="blue", block=False):
    if color == "blue":
        plt.scatter(component_1, component_2, color='blue', label='Original Data', s=50, alpha=0.6)
    else:
        plt.scatter(component_1, component_2, color='red', label='Selected Points', s=100, edgecolor='black')
    plt.xlabel(f" Component 1")
    plt.ylabel(f" Component 2")
    # plt.title(f"{model} of {dataset} Images")
    plt.show(block=block)

def plot_3D(component_1, component_2, component_3, block=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(component_1, component_2, component_3, color='red', label='Selected Points', s=100, edgecolor='black')
    ax.set_xlabel(f"Component 1")
    ax.set_ylabel(f"Component 2")
    ax.set_zlabel(f"Component 3")
    # ax.set_title(f"{model} of {dataset} Images")
    ax.tick_params(color="white", labelcolor="white")
    plt.show(block=block)


############################################################
############################################################
# MAIN
############################################################
############################################################

random_state = 11
random.seed(random_state)
X= [
    sklearn.datasets.make_s_curve(200, random_state=random_state)[0] + [5, 0, 0],
    (sklearn.datasets.make_swiss_roll(200, random_state=random_state)[0] * 0.1) + np.array([5, 5, 5]),
    np.array([[
        random.normalvariate(0, 1),
        random.normalvariate(0, 0.5),
        0
        ] for _ in range(200)])
]


# X[2] = np.hstack((X[2], np.zeros((len(X[2]), 1)) ))
X_stacked = np.vstack([X_i for X_i in X])

for i in range(len(X)):
    X[i] = X[i] - X_stacked.mean(0)
X_stacked = X_stacked - X_stacked.mean(0)


############################################################
# Inter-Component Connections
############################################################
C = inter_connections(X)







############################################################
# Component-wise linearisation
############################################################

S = []
for i in range(len(X)):
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(X[i][:, 0], X[i][:, 1], X[i][:, 2], color='blue', label='Original Data', alpha=0.6)
    # ax.set_xlabel(f"Component 1")
    # ax.set_ylabel(f"Component 2")
    # ax.set_zlabel(f"Component 3")
    # # ax.set_title(f"{model} of {dataset} Images")
    # ax.tick_params(color="white", labelcolor="white")
    # plt.show(block=True)

    
    # nn = 10
    # NG = NearestNeighbors(n_neighbors= nn+1).fit(X[i])
    # NG = NG.kneighbors_graph(X[i]).todense()
    # NG = np.array(NG)
    # NG = NG - np.identity(len(NG))
    
    
    # K = mvu_concurrent(X[i], NG)
    
    # eigenvalues, eigenvectors = np.linalg.eig(K)

    # # take the indices of the greatest eigenvalues
    # # note that the eigenvalues are in a vector (n, 1)
    # n_top_eigenvalues = 3
    # top_eigenvalue_indices = eigenvalues.argsort()[::-1][:n_top_eigenvalues]

    # # get top eigenvalues (n,) and eigenvectors (n, n) - with n small
    # top_eigenvalues = eigenvalues[top_eigenvalue_indices]
    # top_eigenvectors = eigenvectors[:, top_eigenvalue_indices]

    # # diagonalize top eigenvalue vector to (n, n) matrix
    # top_eigenvalue_diag = np.diag(top_eigenvalues ** 0.5)
    # # compute embedded data = Q * lambda^(1/2)
    # S_i = np.dot(top_eigenvectors, top_eigenvalue_diag)
    # S.append(S_i)

    # print(eigenvalues)
    # plot = plt.plot(list(range(len(eigenvalues))),eigenvalues)
    # plot.show()

    isomap = Isomap(n_neighbors=4, n_components=2)
    S_i = isomap.fit_transform(X[i])
    S.append(S_i)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(S_i[:, 0], S_i[:, 1], S_i[:, 2], color='blue', label='Original Data', alpha=0.6)
    # ax.set_xlabel(f"Component 1")
    # ax.set_ylabel(f"Component 2")
    # ax.set_zlabel(f"Component 3")
    # # ax.set_title(f"{model} of {dataset} Images")
    # ax.tick_params(color="white", labelcolor="white")
    # plt.show(block=True)
    # plot_2D(S_i[:, 0], S_i[:, 1], "blue", True)



############################################################
# Inter-Component Connections
############################################################
R = inner_connections(S, C)


############################################################
############################################################
# Reduction
############################################################
############################################################

print(C, R)

reduction = []
for i in range(len(S)):
    for j in range(len(C[i])):
        if i != j and int(C[i][j]) not in reduction:
            reduction.append(int(np.where(np.all(X_stacked == X[i][int(C[i][j])], axis=1))[0]))

    for k in range(len(R[i])):
        if int(R[i][k]) not in reduction:
            reduction.append(int(np.where(np.all(X_stacked == X[i][int(R[i][k])], axis=1))[0]))

reduction.sort()




############################################################
# Connection Graph
############################################################

S_stacked = np.vstack([S_i for S_i in S])

XR_stacked_length = sum([len(S[i][0]) + len(R[i]) for i in range(len(S))])
G = np.zeros((XR_stacked_length, XR_stacked_length))
for i in range(len(S)):
    for j in range(i+1, len(C[i])):
        a = np.where(np.all(X_stacked == X[i][int(C[i][j])], axis=1))[0]
        a_idx = reduction.index(a)

        b = np.where(np.all(X_stacked == X[j][int(C[j][i])], axis=1))[0]
        b_idx = reduction.index(b)
        
        print("inter-components", int(a_idx), int(b_idx))
        
        G[a_idx, b_idx] = (X_stacked[a_idx] - X_stacked[b_idx]) @ (X_stacked[a_idx] - X_stacked[b_idx]).T

    for j in range(len(C[i])):
        if i != j:
            for k in range(j+1, len(C[i])):
                if i != k:
                    a = np.where(np.all(X_stacked == X[i][int(C[i][j])], axis=1))[0]
                    a_idx = reduction.index(a)
                    
                    b = np.where(np.all(X_stacked == X[i][int(C[i][k])], axis=1))[0]
                    b_idx = reduction.index(b)

                    print("intra inter-points", int(a_idx), int(b_idx))
                
                    G[a_idx, b_idx] = (S_stacked[a_idx] - S_stacked[b_idx]) @ (S_stacked[a_idx] - S_stacked[b_idx]).T

    for j in range(len(C[i])):
        if i != j:
            for k in range(len(R[i])):
                a = np.where(np.all(X_stacked == X[i][int(C[i][j])], axis=1))[0]
                a_idx = reduction.index(a)
                
                b = np.where(np.all(X_stacked == X[i][int(R[i][k])], axis=1))[0]
                b_idx = reduction.index(b)

                print("intra-points", int(a_idx), int(b_idx))
                G[a_idx, b_idx] = (S_stacked[a_idx] - S_stacked[b_idx]) @ (S_stacked[a_idx] - S_stacked[b_idx]).T
    if i == 2:
        break


print("max d:", [len(S[i][0]) for i in range(len(X))])
XR_max_d = max([len(S[i][0]) for i in range(len(X))])
XR = [np.zeros((len(S[i][0]) + len(R[i]), XR_max_d)) for i in range(len(S))]
print(len(XR[0][0]))
print(reduction)
XR_idx = [0 for _ in range(len(S))]
for idx in reduction:
    print(idx)
    for i in range(len(S)):
        if idx < len(S[i]):
            temp_point = S[i][idx]
            if len(temp_point) < len(XR[i][XR_idx[i]]):
                temp_point = np.hstack([temp_point, 0])
            XR[i][XR_idx[i]] = temp_point
            XR_idx[i] += 1
            print("found:", i)
            break
        else:
            idx -= len(S[i])
            print("reduced:", idx)
XR_stacked = np.vstack([XR_i for XR_i in XR])

# print(XR)
# print(XR_stacked)
# print(G)



############################################################
############################################################
# Plot Global Structure
############################################################
############################################################

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X_stacked[:, 0], X_stacked[:, 1], X_stacked[:, 2], color='blue', label='Original Data', alpha=0.6)
# ax.scatter(XR[:, 0], XR[:, 1], XR[:, 2], color='red', label='Selected Points', s=100, edgecolor='black')
# for a_idx in range(len(G)):
#     for b_idx in range(len(G[a_idx])):
#         if G[a_idx][b_idx] != 0:

#             a, b = XR[a_idx], XR[b_idx]
#             ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], c="red")
# ax.set_xlabel(f"Component 1")
# ax.set_ylabel(f"Component 2")
# ax.set_zlabel(f"Component 3")
# # ax.set_title(f"{model} of {dataset} Images")
# ax.tick_params(color="white", labelcolor="white")
# plt.show(block=False)







############################################################
# Apply Global MVU
############################################################

K = mvu_global(XR, G)

eigenvalues, eigenvectors = np.linalg.eig(K)

# take the indices of the greatest eigenvalues
# note that the eigenvalues are in a vector (n, 1)
n_top_eigenvalues = 3
top_eigenvalue_indices = eigenvalues.argsort()[::-1][:n_top_eigenvalues]

# get top eigenvalues (n,) and eigenvectors (n, n) - with n small
top_eigenvalues = eigenvalues[top_eigenvalue_indices]
top_eigenvectors = eigenvectors[:, top_eigenvalue_indices]

# diagonalize top eigenvalue vector to (n, n) matrix
top_eigenvalue_diag = np.diag(top_eigenvalues ** 0.5)
# compute embedded data = Q * lambda^(1/2)
embedded_data = np.dot(top_eigenvectors, top_eigenvalue_diag)

# print(embedded_data)

############################################################
# Plot Global MVU
############################################################

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# # ax.scatter(X_stacked[:, 0], X_stacked[:, 1], X_stacked[:, 2], color='blue', label='Original Data', alpha=0.6)
# ax.scatter(embedded_data[:, 0], embedded_data[:, 1], embedded_data[:, 2], color='red', label='Selected Points', s=100, edgecolor='black')

# for a_idx in range(len(G)):
#     for b_idx in range(len(G[a_idx])):
#         if G[a_idx][b_idx] != 0:
#             a, b = embedded_data[a_idx], embedded_data[b_idx]
#             ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], c="red")

# ax.set_xlabel(f"Component 1")
# ax.set_ylabel(f"Component 2")
# ax.set_zlabel(f"Component 3")
# # ax.set_title(f"{model} of {dataset} Images")
# ax.tick_params(color="white", labelcolor="white")
# plt.show(block=True)



for i in range(len(XR)):
    for j in range(len(XR[i])):
        global_idx = sum([len(XR[k]) for k in range(i)]) + j
        print(i,j,global_idx, XR_stacked[global_idx], XR[i][j])






