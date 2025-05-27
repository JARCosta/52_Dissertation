import datetime
import json
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

def warning(text:str):
    print(f"{bcolors.ORANGE}Warning: {text}{bcolors.ENDC}")

def important(text:str):
    print(f"{bcolors.BLUE}{text}{bcolors.ENDC}")

############################################################
# MEASURES #################################################
############################################################

def store_measure(model_args:dict, One_nn:float=None, T:float=None, C:float=None, fail=None, best=False):

    def measure_json(model_args:dict, best:bool=False):
        try:
            if best:
                with open("measures.best.json", "r") as f:
                    measures = json.loads(f.read())
            else:
                with open("measures.all.json", "r") as f:
                    measures = json.loads(f.read())
        except FileNotFoundError:
            measures = {}
        measures[model_args['dataname']] = measures.get(model_args['dataname'], {})
        measures[model_args['dataname']][model_args['model']] = measures[model_args['dataname']].get(model_args['model'], {})
        if not best:
            measures[model_args['dataname']][model_args['model']][model_args['#neighs']] = measures[model_args['dataname']][model_args['model']].get(model_args['#neighs'], {})
        return measures

    
    model_args['end'] = datetime.datetime.now().strftime("%d-%H:%M:%S")
    data = {
        'start': model_args.get('start', ""),
        'end': model_args.get('end', ""),
        'paper': model_args.get('paper', ""),
        'dataname': model_args.get('dataname', ""),
        'model': model_args.get('model', ""),
        '#points': model_args.get('#points', ""),
        '#neighs': model_args.get('#neighs', ""),
    }
    if fail is not None:
        data['1-NN'] = ""
        data['T'] = ""
        data['C'] = ""
        data['fail'] = fail
        warning(f"storing failed model ({fail})")
    else:
        data['1-NN'] = One_nn if One_nn is not None else ""
        data['T'] = T
        data['C'] = C
    if best:
        data.pop('start')
        data.pop('end')
        with open("measures.best.csv", "a") as f:
            f.write(str(list(data.values())).replace("'", "").replace("[", "").replace("]", "").replace(", ", ",") + "\n")
        
        measures = measure_json(model_args, best)
        measures[model_args['dataname']][model_args['model']] = {'#neighs': model_args['#neighs'], '#points': model_args['#points'], '1-NN': One_nn, 'T': T, 'C': C}
        with open("measures.best.json", "w") as f:        
            f.write(json.dumps(measures, indent=4) + "\n")
    else:
        data.pop('paper')
        with open("measures.all.csv", "a") as f:
            f.write(str(list(data.values())).replace("'", "").replace("[", "").replace("]", "").replace(", ", ",") + "\n")
        
        measures = measure_json(model_args, best)
        measures[model_args['dataname']][model_args['model']][model_args['#neighs']] = {'#points': model_args['#points'], '1-NN': One_nn, 'T': T, 'C': C}
        with open("measures.all.json", "w") as f:        
            f.write(json.dumps(measures, indent=4) + "\n")




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


