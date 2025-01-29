
import random
import sys

import cvxpy as cvx
import numpy as np
from scipy.io import loadmat

import plotter
import mvus


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

if __name__ == '__main__':

    dataset = "teapots"
    try:
        data = np.load(f'datasets/{dataset}.npy')
    except FileNotFoundError:
        data = loadmat(f'datasets/{dataset}.mat')
        data = data['Input'][0][0][0]
        data = data.T
        data = data.astype(np.float64)

    print(data.shape)
    print("Total Images:",data.shape[0])
    print("Pixels in each RGB Image: 76x101x3 =",data.shape[1])

    print(cvx.installed_solvers())

    

    eps = 1e-3
    n_neighbors = 3
    if len(sys.argv) > 1:
        model = sys.argv[1]
        if model == "mvu":
            Gram = mvus.mvu(data, eps=eps, n_neighbors=n_neighbors)
        elif model == "mvu_n_by_k":
            Gram = mvus.mvu_n_by_k(data, eps=eps, n_neighbors=n_neighbors)
        elif model == "mvu_no_sdp":
            Gram = mvus.mvu_no_sdp(data, eps=eps, n_neighbors=n_neighbors)
        elif model == "mvu_no_sdp_ori":
            Gram = mvus.mvu_no_sdp_ori(data, eps=eps, n_neighbors=n_neighbors)
        else:
            print("Invalid model")
            sys.exit(1)
    else:
        Gram = mvus.mvu(data, eps=eps, n_neighbors=n_neighbors)
    print(Gram == 0, (Gram == 0).any())
    if Gram is not None and (Gram == 0).any():
        plotter.save_gram(Gram, f"resources/{dataset}.{eps:.0e}.{n_neighbors}.gram")
        plotter.plot_MVU(Gram, dataset)
