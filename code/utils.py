import datetime
import os
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

class bcolors:
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    REVERSE = '\033[7m'

    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    ORANGE = '\033[33m'
    BLUE = '\033[34m'
    PURPLE = '\033[35m'
    CYAN = '\033[36m'
    LIGHTGRAY = '\033[37m'

    DARKGRAY = '\033[90m'
    LIGHTRED = '\033[91m'
    LIGHTGREEN = '\033[92m'
    YELLOW = '\033[93m'
    LIGHTBLUE = '\033[94m'
    PINK = '\033[95m'
    LIGHTCYAN = '\033[96m'


class Stamp:
    def __init__(self):
        self.timestamp = datetime.datetime.now()
        self.color = bcolors.GREEN

    def set(self):
        self.timestamp = datetime.datetime.now()

    def print(self, text:str="*"):
        time_diff = datetime.datetime.now() - self.timestamp
        if time_diff.total_seconds() > 1:
            self.color = bcolors.RED
        else:
            self.color = bcolors.GREEN
        minutes = time_diff.seconds // 60
        seconds = time_diff.seconds % 60
        milliseconds = time_diff.microseconds // 1000
        time_diff = '{:02}:{:02}.{:02}'.format(minutes, seconds, milliseconds)
        print(text.replace("*", self.color + time_diff + bcolors.ENDC ))
        return self
    
    def print_set(self, text:str="*"):
        return self.print(text).set()

stamp = Stamp()

############################################################
# MEASURES #################################################
############################################################

def store_measure(model_args:dict, One_nn:float=None, T:float=None, C:float=None, fail=None):
    model_args['end'] = datetime.datetime.now().strftime("%d-%H:%M:%S")
    data = {
        'start': model_args['start'],
        'end': model_args['end'],
        'dataname': model_args['dataname'],
        'model': model_args['model'],
        '#points': model_args['#points'],
        '#neighs': model_args['#neighs'],
    }
    if fail is not None:
        data['1-NN'] = None
        data['T'] = None
        data['C'] = None
        data['fail'] = fail
        print(f"Warning: storing failed model ({fail})")
    else:
        data['1-NN'] = One_nn
        data['T'] = T
        data['C'] = C
    with open("measures.all.csv", "a") as f:
        f.write(str(list(data.values())).replace("'", "").replace("[", "").replace("]", "").replace(", ", ",") + "\n")


############################################################
# CACHE ####################################################
############################################################

def save_cache(model_args:dict, data, datatype:str):
    model_args_copy = model_args.copy()
    if "model" in model_args.keys():
        file_name = model_args["model"] + "/"
        model_args_copy.pop('model')
    else:
        file_name = ""
    file_name += str(list(model_args_copy.values())).replace("'", "").replace("[", "").replace("]", "").replace(", ", ".") + f".{datatype}"
    # print(file_name)

    import pickle
    if not os.path.exists("cache"):
        os.makedirs("cache")
    if not os.path.exists(f"cache/{model_args['model']}"):
        os.makedirs(f"cache/{model_args['model']}")
    with open(f"cache/{file_name}.out", "wb") as f:
        pickle.dump(data, f)


def load_cache(model_args:dict, datatype:str):
    model_args_copy = model_args.copy()
    if 'model' in model_args.keys():
        file_name = model_args["model"] + "/"
        model_args_copy.pop('model')
    else:
        file_name = ""
    file_name += str(list(model_args_copy.values())).replace("'", "").replace("[", "").replace("]", "").replace(", ", ".") + f".{datatype}"
    # print(f"cache/{file_name}")

    import pickle
    try:
        with open(f"cache/{file_name}.out", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None
    


def intra_dist_matrix(data: np.ndarray):
    n_samples = data.shape[0]
    dist, indices = NearestNeighbors(n_neighbors=n_samples).fit(data).kneighbors(data)
    dist_matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        dist_matrix[i][indices[i]] = dist[i]
    return dist_matrix





def k_neigh(data: np.ndarray, n_neighbors: int, reduction:bool=None):
    n = data.shape[0]

    K_neigh = NearestNeighbors(n_neighbors=n_neighbors+1)
    K_neigh.fit(data)

    # Get nearest neighbor indices (excluding self)
    distances, indices = K_neigh.kneighbors(data)
    Neigh_distances = distances[:, 1:]  # Remove the first column (self-index)
    Neigh_indices = indices[:, 1:]  # Remove the first column (self-index)

    Neigh_matrix = np.zeros((n,n))
    for i in range(n):
        Neigh_matrix[i][Neigh_indices[i]] = Neigh_distances[i]

    if reduction != None:
        Neigh_matrix = np.maximum(Neigh_matrix, Neigh_matrix.T)
        if reduction:
            Neigh_matrix = np.triu(Neigh_matrix)
        # print("NM size:", np.count_nonzero(Neigh_matrix))
    Neigh_graph = csr_matrix(Neigh_matrix)  # Convert back to sparse format

    return Neigh_graph, Neigh_matrix, Neigh_indices

def connected(NM:np.ndarray):
    NM_copy = NM.copy()

    class bfs:

        def __init__(self, graph_matrix):
            import random
            self.graph_matrix = graph_matrix
            self.visited = set()
            self.nodes_left = set(range(graph_matrix.shape[0]))
            self.to_visit = set((random.choice(list(self.nodes_left)),))
        
        def visit(self, node):
            # print(f"Visiting {node}, {len(self.nodes_left)} nodes left")
            self.visited.add(node)
            self.nodes_left.remove(node)
            for neighbor in np.where(self.graph_matrix[node] != 0)[0]:
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
                    neis = np.where(self.graph_matrix[node] != 0)[0]
                    for nei in neis:
                        if nei in self.visited:
                            self.visit(node)
                            added += 1
                            break
            return len(self.nodes_left) == 0, self.nodes_left

    for i in range(NM_copy.shape[0]):
        for j in range(NM_copy.shape[1]):
            if NM_copy[i][j] != 0 and NM_copy[j][i] != 0:
                NM_copy[j][i] = 0
    connected, unvisited = bfs(NM_copy + NM_copy.T).connected()
    if not connected:
        # print(unvisited)
        # print(f"ERROR: couldn't connect {len(unvisited)} out of the {NM.shape[0]} total points.")
        return False
        raise Exception(f"ERROR: couldn't connect {len(unvisited)} out of the {NM.shape[0]} total points")
    return True









def embed(K, reduce:bool=True, verbose:bool=True):
    # eigenvalues, eigenvectors = np.linalg.eigh(K)
    eigenvalues, eigenvectors = np.linalg.eig(K)

    idx = np.argsort(eigenvalues)[::-1]  # Sort eigenvalues in descending order
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    eigenvalues = np.sqrt(np.abs(eigenvalues))

    if verbose:
        print("eigenvalues:\n", list(eigenvalues))
        print("std:", np.std(eigenvalues))
        print("median:", np.median(eigenvalues))

    eigenvalues_idx = [i for i in range(len(eigenvalues)) if eigenvalues[i] > (np.median(eigenvalues) + np.std(eigenvalues))]
    if reduce:
        eigenvalues = eigenvalues[eigenvalues_idx]
        eigenvectors = eigenvectors[:, eigenvalues_idx]

    Y = eigenvectors @ np.diag(eigenvalues)

    if verbose:
        print("filtered eigenvalues:\n", list(eigenvalues))
        print("filtered eigenvectors:\n", list(eigenvectors)[0])

    return Y

def fast_embed(K, num_eigen:int, verbose:bool=True):
    from scipy.sparse.linalg import eigsh  # Sparse eigen decomposition

    eigenvalues, eigenvectors = eigsh(K, k=num_eigen, which='LM')  # Faster than full eigen decomposition

    idx = np.argsort(eigenvalues)[::-1]  # Sort eigenvalues in descending order
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    eigenvalues = np.sqrt(np.abs(eigenvalues))

    if verbose:
        print("filtered eigenvalues:\n", list(eigenvalues), eigenvalues.shape)
        print("filtered eigenvectors:\n", list(eigenvectors), eigenvectors.shape)

    # eigenvalues = np.flipud(eigenvalues)
    # eigenvectors = np.flipud(eigenvectors)

    eigenvalues = np.diag(np.sqrt(np.maximum(eigenvalues, 0)))  # Avoid negative values
    Y = eigenvectors @ eigenvalues  # Compute lower-dimensional representation

    return Y


