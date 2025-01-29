import random
import matplotlib.pyplot as plt
from scipy.io import loadmat

import mvu_aux



# applying PCA
from sklearn.decomposition import PCA
def pca(X): 
    pca = PCA(n_components=2)
    pca.fit(X.T)
    teapot_pca = pca.transform(X.T)

    mvu_aux.plot_2D(teapot_pca[:,0],teapot_pca[:,1],'PCA', "Teapot")

# applying t-SNE
from sklearn.manifold import TSNE
def tsne(X):
    tsne = TSNE(n_components=2)
    teapot_tsne = tsne.fit_transform(X.T)

    mvu_aux.plot_2D(teapot_tsne[:,0],teapot_tsne[:,1],'t-SNE', "Teapot")

# applying ISOMAP
from sklearn.manifold import Isomap
def isomap(X):
    isomap = Isomap(n_components=3, n_neighbors=10)
    teapot_isomap = isomap.fit_transform(X.T)

    mvu_aux.plot_2D(teapot_isomap[:,0],teapot_isomap[:,1],'ISOMAP', "Teapot")
    mvu_aux.plot_3D(teapot_isomap[:,0],teapot_isomap[:,1],teapot_isomap[:,2],'ISOMAP', "Teapot")

# applying MVU
import cvxpy as cvx
import numpy as np
from sklearn.neighbors import NearestNeighbors


class bfs:

    def __init__(self, graph_matrix):
        self.graph_matrix = graph_matrix
        self.visited = set()
        self.nodes_left = set(range(graph_matrix.shape[0]))
        self.to_visit = set((random.choice(list(self.nodes_left)),))
    
    def visit(self, node):
        # print(f"Visiting {node}, {len(self.nodes_left)} nodes left")
        self.visited.add(node)
        self.nodes_left.remove(node)
        for neighbor in np.where(self.graph_matrix[node] == 1)[0]:
            if neighbor not in self.visited:
                self.to_visit.add(neighbor)
            else:
                pass
                # print(f"Node {neighbor} already visited")
    
    def connected(self):
        while self.to_visit:
            self.to_visit = set(sorted(list(self.to_visit)))
            # print(self.to_visit)
            self.visit(self.to_visit.pop())
        
        # check for connected nodes that are not referenced
        added = 1
        while added != 0:
            added = 0
            for node in list(self.nodes_left):
                neis = np.where(self.graph_matrix[node] == 1)[0]
                for nei in neis:
                    if nei in self.visited:
                        self.visit(node)
                        added += 1
                        break
        return len(self.nodes_left) == 0, self.nodes_left

def callback(data):
    # Example of accessing information in the callback
    # print(f"Iteration: {data['info']['iter']}, Objective Value: {data['info']['pobj']}")
    print(data)
    # Gram = data['info']['pobj']
    # if Gram is not None and Gram.sum() != 0:
    #     print(f"Saving Gram at iter={data['info']['iter']}")
    #     mvu_aux.save_gram(Gram, f"resources/{dataset.split('.')[0]}.{eps:.0e}.{n_neighbors}.gram.npy")
    #     mvu_aux.plot_MVU(Gram, dataset, block=False)

def MVU(X, eps=1e-4, n_neighbors=4):

    n_points = X.shape[0]
    # mvu_aux.plot_Gram((X @ X.T) * 1e-6, "Teapot")


    neighbors = NearestNeighbors(n_neighbors=n_neighbors + 1)
    neighbors.fit(X)
    neighbors = neighbors.kneighbors_graph(X).todense()
    neighbors = np.array(neighbors)
    connections = neighbors - np.eye(n_points)

    with open("resources/connections.txt", "w") as f:
        for i in connections:
            f.write(str(list(i)).replace("\n","") + "\n")



    # fractions = 1
    # print([[i * connections.shape[0]//fractions ,(i+1) * connections.shape[0]//fractions] for i in range(fractions)])
    # [mvu_aux.plot_connections(connections[i * connections.shape[0]//fractions : (i+1) * connections.shape[0]//fractions], n_neighbors=n_neighbors) for i in range(fractions)]
    
    
    # nn_temp = neighbors #/ (n_neighbors+1)
    with open("resources/neighbors.txt", "w") as f:
        for i in range(n_points):
            f.write(str(connections[i]).replace("\n","") + "\n")
    with open("resources/neighbors.connections.txt", "w") as f:
        for i in range(n_points):
            # write the node's id and its neighbors
            f.write(f"{i}: {list(np.where(connections[i] == 1)[0])}\n")
    
    # print("Number of Neighbors:", np.mean(np.sum(neighbors, axis=1), axis=0))
    # with open("resources/rec_neighbors.txt", "w") as f:
    #     # for i in range(n_points):
    #     #     f.write(str(neighbors[i]).replace("\n","") + "\n")
    #     temp = np.eye(n_points)
    #     for _ in range(1):
    #         temp = temp @ nn_temp
    #     for i in range(n_points):
    #         f.write(str(temp[i]).replace("\n","") + "\n")

    connected, unvisited = bfs(connections).connected()

    if not connected:
        print(f"ERROR: couldn't connect {len(unvisited)} out of the {n_points} total points for {n_neighbors} neighbors")
        print(unvisited)
        return None

    # inner product matrix of the original data
    X = cvx.Constant((X @ X.T))

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
            if j_id != 0:
                constraints.append(G[i, 0] >= G[i, j_id])
        sum = cvx.sum(G[i][1:])
        constraints.append(G[i, 0] >= sum)
        constraints.append(G[i, 0] >= 1e-8)
    print("Length of constraints:",len(constraints))

    try:
        # save_problem(objective, constraints, f"resources/{dataset.split('.')[0]}.{eps:.0e}.{n_neighbors}.problem.txt")
        problem = cvx.Problem(objective, constraints)
        problem.solve(verbose=True, eps=eps)


    except (KeyboardInterrupt, cvx.error.SolverError) as e:
        print("Interrupted")
        print(e) # TODO: add traceback
        pass

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



if __name__ == '__main__':

    dataset = "teapots"
    try:
        data = np.load(f'datasets/{dataset}.npy')
    except FileNotFoundError:
        data = loadmat(f'datasets/{dataset}.mat')
        data = data['Input'][0][0][0]
        data = data.T
        data = data.astype(np.float64)
        # cut the dataset to 100 images
        # data = data[3:]
    # if dataset == "umist":
        
    with open(f"resources/X_{dataset}.txt", "w") as f:
        # for i in data:
        #     for j in i:
        #         f.write(str(j).replace("\n","") + "\n")
        for i in data:
            f.write(str(i).replace("\n","") + "\n")
        

    print(data.shape)
    print("Total Images:",data.shape[0])
    print("Pixels in each RGB Image: 76x101x3 =",data.shape[1])

    # pca(data)
    # tsne(data)
    # isomap(data)
    print(cvx.installed_solvers())

    eps = 1e-8
    n_neighbors = 3
    Gram = MVU(data, eps=eps, n_neighbors=n_neighbors)
    if Gram is not None and Gram.sum() != 0:
        mvu_aux.save_gram(Gram, f"resources/{dataset}.{eps:.0e}.{n_neighbors}.gram.npy")
        mvu_aux.plot_MVU(Gram, dataset)




