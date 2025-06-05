import datetime
import json
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

def warning(text:str):
    print(f"{bcolors.ORANGE}Warning: {text}{bcolors.ENDC}")

def important(text:str):
    print(f"{bcolors.BLUE}{text}{bcolors.ENDC}")

############################################################
# K-NEIGHBOURHOOD ##########################################
############################################################


def k_neigh(data, n_neighbors, bidirectional=False, common_neighbors=False):
    """
    Builds a k-nearest neighbor (k-NN) neighborhood matrix.

    Args:
        data (numpy.ndarray): The dataset (n_samples, n_features).
        bidirectional (bool, optional): If True, the connections are bidirectional, neighbourhood matrix is upper triangular.
            Defaults to False, neighbourhood matrix is directly computed from the k-nn graph.
        common_neighbors (bool, optional): If True, considers nodes with common neighbours to be neighbours as well.
            Defaults to False.

    Returns:
        numpy.ndarray: The neighborhood matrix (n_samples, n_samples).
    """
    k = n_neighbors
    n_samples = data.shape[0]
    dist_matrix = intra_dist_matrix(data[:, :5]) # TODO: I don't get how can sklearn.neighbors.NearestNeighbors be so much faster than scipy.spatial.distance.cdist, it knows its cdist between the same data?
    neigh_matrix = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        # Find indices of k nearest neighbors (excluding itself)
        nearest_indices = np.argsort(dist_matrix[i])[0:k + 1]
        
        # Remove itself from the list of nearest neighbors
        nearest_indices = np.delete(nearest_indices, np.where(nearest_indices == i))
        nearest_indices = nearest_indices[0:k]

        for j in nearest_indices:
            neigh_matrix[i, j] = dist_matrix[i, j] if dist_matrix[i, j] > 0 else 1e-10

    print(f"connections: {np.count_nonzero(neigh_matrix)}")
    if bidirectional:
        neigh_bidirectional = np.maximum(neigh_matrix, neigh_matrix.T) # two-way unidirectional connections
        neigh_matrix = np.triu(neigh_bidirectional) # keep only one connection per pair (upper triangle)

    print(f"connections: {np.count_nonzero(neigh_matrix)}")
    if common_neighbors: # if i and j are my neighs, then i and j are neighs of each other
        adj_bool = neigh_matrix > 0
        common_neigh = adj_bool.astype(int) @ adj_bool.astype(int).T
        common_neigh = (common_neigh > 0) & (~adj_bool) # selection of common neighbors that are not already connected
        for i,j in zip(*np.where(common_neigh)):
            shortest_path = np.inf
            # Find the k node that minimizes the distance between i and j
            for k in range(n_samples):
                if neigh_matrix[i,k] > 0 and neigh_matrix[j,k] > 0:
                    shortest_path = min(shortest_path, neigh_matrix[i,k] + neigh_matrix[j,k])
            neigh_matrix[i,j] = shortest_path
            neigh_matrix[j,i] = shortest_path
    print(f"connections: {np.count_nonzero(neigh_matrix)}")
    return neigh_matrix

def k_graph(neigh_matrix:np.ndarray):
    return csr_matrix(neigh_matrix != 0)

############################################################
# MEASURES #################################################
############################################################

def TC(X, Y, n_neighbors) -> tuple[float, float]:
    
    n_samples = X.shape[0]
    
    NM_X = k_neigh(X, n_neighbors) # need symetrix
    NM_Y = k_neigh(Y, n_neighbors) # need symetrix

    NM_X = np.where(NM_X != 0, 1, 0)
    NM_Y = np.where(NM_Y != 0, 1, 0)

    NM_T = NM_Y - NM_X

    NM_T[np.where(NM_T == -1)] = 0
    D_X = intra_dist_matrix(X)
    R_X = np.argsort(np.argsort(D_X, axis=1), axis=1)
    T = NM_T * R_X
    T[T != 0] -= n_neighbors
    T = 1 - np.sum(T) * (2 / (n_samples * n_neighbors * (2*n_samples - 3*n_neighbors - 1)))



    NM_C = NM_X - NM_Y
    NM_C[np.where(NM_C == -1)] = 0
    D_Y = intra_dist_matrix(Y)
    R_Y = np.argsort(np.argsort(D_Y, axis=1), axis=1)
    C = NM_C * R_Y
    C[C != 0] -= n_neighbors
    C = 1 - np.sum(C) * (2 / (n_samples * n_neighbors * (2*n_samples - 3*n_neighbors - 1)))

    return round(float(T), 3), round(float(C), 3)

def  one_NN(Y, labels) -> float:
    if labels is None:
        warning("labels is None")
        return None
    
    NM = k_neigh(Y, 1) # need symetrix

    Y_labels = np.zeros(labels.shape)
    for i in range(Y.shape[0]):
        Y_labels[i] = labels[np.where(NM[i] != 0)]
    
    one_NN = np.count_nonzero(Y_labels - labels) / labels.shape[0]
    return round(float(one_NN), 3)





def get_measures(X, Y, labels, n_neigh) -> tuple[float, float, float]:
    
    stamp.set()
    One_nn = one_NN(Y, labels)
    T, C = TC(X, Y, n_neigh)
    stamp.print_set(f"*\t 1_NN, T, C \t {One_nn}, {T}, {C}")

    return One_nn, T, C

def store_measure(model_args:dict, One_nn:float=None, T:float=None, C:float=None, fail=None, best=False):
    def get_json(dataname:str, model:str, n_neighs:int, best:bool=False):
        try:
            if best:
                with open("measures.best.json", "r") as f:
                    measures = json.loads(f.read())
            else:
                with open("measures.all.json", "r") as f:
                    measures = json.loads(f.read())
        except (FileNotFoundError, json.JSONDecodeError):
            measures = {}
        measures[dataname] = measures.get(dataname, {})
        measures[dataname][model] = measures[dataname].get(model, {})
        if not best:
            measures[dataname][model][str(n_neighs)] = measures[dataname][model].get(str(n_neighs), {})
        return measures

    dataname = model_args['dataname']
    model = model_args['model']
    n_neighs = model_args['#neighs']
    points = model_args['#points']



    if best:
        measures = get_json(dataname, model, n_neighs, best)
        measures[dataname][model] = {'#neighs': n_neighs, '#points': points, '1-NN': One_nn, 'T': T, 'C': C}
        with open("measures.best.json", "w") as f:
            f.write(json.dumps(measures, indent=4) + "\n")
    else:
        measures = get_json(dataname, model, n_neighs, best)
        measures[dataname][model][str(n_neighs)] = {'#points': points, '1-NN': One_nn, 'T': T, 'C': C, 'status': model_args['status']}
        with open("measures.all.json", "w") as f:
            f.write(json.dumps(measures, indent=4) + "\n")

    json_to_csv(best=True)
    json_to_csv(best=False)

def json_to_csv(best:bool):
    with open(f"measures.{'best' if best else 'all'}.json", "r") as f:
        data = json.load(f)
    
    with open(f"measures.{'best' if best else 'all'}.csv", "w") as f:
        f.write(f"dataname,model,n_neighs,points,1-NN,T,C\n")
        if best:
            for dataname, model in data.items():
                for model, data in model.items():
                    f.write(f"{dataname},{model},{data['#neighs']},{data['#points']},{data['1-NN']},{data['T']},{data['C']}\n")
        else:
            for dataname, model in data.items():    
                for model, n_neighs in model.items():
                    for n_neighs, data in n_neighs.items():
                        f.write(f"{dataname},{model},{n_neighs},{data['#points']},{data['1-NN']},{data['T']},{data['C']}\n")






    


def intra_dist_matrix(data: np.ndarray):
    n_samples = data.shape[0]
    dist, indices = NearestNeighbors(n_neighbors=n_samples).fit(data).kneighbors(data)
    dist_matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        dist_matrix[i][indices[i]] = dist[i]
    return dist_matrix
