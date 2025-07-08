
import random
import matplotlib.pyplot as plt
from scipy.io import loadmat
import traceback
import cvxpy as cvx
import numpy as np
from sklearn.neighbors import NearestNeighbors

from main import bfs

def mvu(X, eps=1e-4, n_neighbors=4):

    n_points = X.shape[0]

    neighbors = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X)
    neighbors = neighbors.kneighbors_graph(X).todense()
    neighbors = np.array(neighbors)
    connections = neighbors - np.eye(n_points)

    connected, unvisited = bfs(connections).connected()
    if not connected:
        print(f"ERROR: couldn't connect {len(unvisited)} out of the {n_points} total points for {n_neighbors} neighbors")
        print(unvisited)
        return None

    # inner product matrix of the original data
    X = cvx.Constant((X @ X.T) * 1e-6)

    # inner product matrix of the projected points; PSD constrains G to be both PSD and symmetric
    G = cvx.Variable((n_points, n_points), PSD=True)
    G.value = np.zeros((n_points, n_points))
    print("G shape:", G.shape)

    # spread out points in target manifold
    objective = cvx.Maximize(cvx.trace(G))
    constraints = [cvx.sum(G) == 0]

    # add distance-preserving constraints
    for i in range(n_points):
        for j in range(n_points):
            if connections[i, j] == 1:
                constraints.append(
                    (X[i, i] - 2 * X[i, j] + X[j, j]) - 
                    (G[i, i] - 2 * G[i, j] + G[j, j]) == 0
                )
        
    print("Length of constraints:",len(constraints))

    problem = cvx.Problem(objective, constraints)
    problem.solve(solver="SCS", verbose=True, eps=eps)
    return G.value

def mvu_n_by_k(X, eps=1e-4, n_neighbors=4):

    n_points = X.shape[0]

    neighbors = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X)
    neighbors = neighbors.kneighbors_graph(X).todense()
    neighbors = np.array(neighbors)
    connections = neighbors - np.eye(n_points)

    connected, unvisited = bfs(connections).connected()
    if not connected:
        print(f"ERROR: couldn't connect {len(unvisited)} out of the {n_points} total points for {n_neighbors} neighbors")
        print(unvisited)
        return None

    # inner product matrix of the original data
    X = cvx.Constant((X @ X.T) * 1e-6)

    # inner product matrix of the projected points; PSD constrains G to be both PSD and symmetric
    G = cvx.Variable((n_points, n_neighbors + 1))
    G.value = np.zeros((n_points, n_neighbors + 1))
    print("G shape:", G.shape)

    # spread out points in target manifold
    objective = cvx.Maximize(cvx.sum(G @ np.array([1,] + [0,] * n_neighbors)))
    constraints = [cvx.sum(G) == 0]

    # add distance-preserving constraints
    for i in range(n_points):
        neis = [i,] + list(np.where(connections[i] == 1)[0])
        for j in neis:
            j_id = neis.index(j)
            constraints.append(
                (X[i, i] - 2 * X[i, j] + X[j, j]) - 
                (G[i, 0] - 2 * G[i, j_id] + G[j, 0]) == 0
            )
        constraints.append(G[i, 0] >= cvx.sum(G[i][1:]))
        constraints.append(G[i, 0] >= 1e-8)
    print("Length of constraints:",len(constraints))

    problem = cvx.Problem(objective, constraints)
    problem.solve(verbose=True, eps=eps)

    # print(G.value)
    # for i in range(n_points):
    #     print(f"Point {i}:", G.value[i])

    gram = np.zeros((n_points, n_points))
    for i in range(n_points):
        neis = [i,] + list(np.where(connections[i] == 1)[0])
        for j in neis:
            j_id = neis.index(j)
            gram[i, j] = G.value[i, j_id]
    
    return gram

def mvu_no_sdp_ori(X, eps=1e-4, n_neighbors=4):

    n_points = X.shape[0]

    neighbors = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X)
    neighbors = neighbors.kneighbors_graph(X).todense()
    neighbors = np.array(neighbors)
    connections = neighbors - np.eye(n_points)

    connected, unvisited = bfs(connections).connected()
    if not connected:
        print(f"ERROR: couldn't connect {len(unvisited)} out of the {n_points} total points for {n_neighbors} neighbors")
        print(unvisited)
        return None

    # inner product matrix of the original data
    X = cvx.Constant((X @ X.T) * 1e-6)

    # inner product matrix of the projected points; PSD constrains G to be both PSD and symmetric
    G = cvx.Variable((n_points, n_points), symmetric=True)
    G.value = np.zeros((n_points, n_points))
    print("G shape:", G.shape)

    # spread out points in target manifold
    objective = cvx.Maximize(cvx.trace(G))
    constraints = [cvx.sum(G) == 0]

    # add distance-preserving constraints
    for i in range(n_points):
        for j in range(n_points):
            if connections[i, j] == 1:
                constraints.append(
                    (X[i, i] - 2 * X[i, j] + X[j, j]) - 
                    (G[i, i] - 2 * G[i, j] + G[j, j]) == 0
                )
            constraints.append(G[i, j] >= 0)
        line = np.ones(n_points)
        line[i] = 0
        constraints.append(G[i, i] >= cvx.sum(G[i] @ line))
    print("Length of constraints:",len(constraints))

    problem = cvx.Problem(objective, constraints)
    problem.solve(solver="CLARABEL", verbose=True)
    return G.value

def mvu_no_sdp(X, eps=1e-4, n_neighbors=4):

    n_points = X.shape[0]

    neighbors = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X)
    neighbors = neighbors.kneighbors_graph(X).todense()
    neighbors = np.array(neighbors)
    connections = neighbors - np.eye(n_points)

    connected, unvisited = bfs(connections).connected()
    if not connected:
        print(f"ERROR: couldn't connect {len(unvisited)} out of the {n_points} total points for {n_neighbors} neighbors")
        print(unvisited)
        return None

    # inner product matrix of the original data
    X = cvx.Constant((X @ X.T) * 1e-6)

    # inner product matrix of the projected points; PSD constrains G to be both PSD and symmetric
    G = cvx.Variable((n_points, n_points))
    G.value = np.zeros((n_points, n_points))
    print("G shape:", G.shape)

    # spread out points in target manifold
    objective = cvx.Maximize(cvx.trace(G))
    constraints = [cvx.sum(G) == 0]

    # add distance-preserving constraints
    for i in range(n_points):
        for j in range(n_points):
            if connections[i, j] == 1:
                constraints.append(
                    (X[i, i] - 2 * X[i, j] + X[j, j]) - 
                    (G[i, i] - 2 * G[i, j] + G[j, j]) == 0
                )
                constraints.append(G[i, j] == G[j, i])
            if i == j:
                constraints.append(G[i, j] >= 0)
        
    print("Length of constraints:",len(constraints))

    problem = cvx.Problem(objective, constraints)
    problem.solve(verbose=True, eps=eps)
    return G.value



