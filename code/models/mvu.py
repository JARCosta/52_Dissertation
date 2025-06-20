import datetime
import numpy as np
import cvxpy as cvx

import mosek
from scipy.sparse import csgraph, csr_matrix
from scipy.spatial.distance import cdist
import matlab.engine
import os

import plot
import utils
import models.neighbourhood
import models.extensions

eng = None

class MVU(models.Neighbourhood):

    def __init__(self, model_args:dict, n_neighbors:int, eps:float=1e-3):
        super().__init__(model_args, n_neighbors)
        self.eps = eps
        self._mode = 0
        
        global eng
        if eng is None:
            eng = matlab.engine.start_matlab()
            # Add the directory containing the MATLAB function to the MATLAB path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            eng.addpath(current_dir, nargout=0)
            utils.stamp.print(f"*\t {self.model_args['model']}\t MATLAB started")

    def _neigh_matrix(self, X:np.ndarray):
        return self.k_neigh(X, bidirectional=True, common_neighbors=False)

    def _fit(self, X:np.ndarray):
        """Fit the MVU model and compute the low-dimensional embeddings using MATLAB."""

        cc, labels = csgraph.connected_components(self.NM, directed=False)
        if cc > 1:
            oldNM = self.NM.copy()
            utils.warning(f"Warning: {cc} components found. Adding shortest connections possible to merge components.")
            self.model_args['artificial_connected'] = True
            while cc > 1:
                largest_component = np.argmax(np.bincount(labels))
                largest_component_idx = np.where(labels == largest_component)[0]
                other_idx = np.where(labels != largest_component)[0]

                distances = cdist(X[largest_component_idx], X[other_idx])
                shortest_distance = np.min(distances)
                ab = np.where(distances == shortest_distance)

                a_idx, b_idx = largest_component_idx[ab[0]], other_idx[ab[1]]
                self.NM[a_idx, b_idx] = np.linalg.norm(X[a_idx] - X[b_idx])

                cc, labels = csgraph.connected_components(self.NM, directed=False)
            if self.model_args['plotation']:
                plot.plot_two(X, X, oldNM, self.NM, False, False, block=False, title=f"{self.model_args['dataname']} {self.model_args['#neighs']} neighbors")

        # Convert data to MATLAB format
        X_matlab = matlab.double(X.tolist())
        
        # Extract index pairs from sparse matrix
        # neigh_graph.nonzero() returns (row_indices, col_indices) of non-zero elements
        rows, cols = utils.k_graph(self.NM).nonzero()
        
        # Convert to list of index pairs and add 1 for MATLAB's 1-based indexing
        neighbor_pairs = [[int(i) + 1, int(j) + 1] for i, j in zip(rows, cols)]

        N_matlab = matlab.double(neighbor_pairs)
        
        utils.stamp.print(f"*\t {self.model_args['model']}\t calling MATLAB")
        start = datetime.datetime.now()
        K_matlab, cvx_status = eng.solve_mvu_optimization(X_matlab, N_matlab, self.model_args['eps'], nargout=2)
        end = datetime.datetime.now()
        print(f"Time taken: {datetime.timedelta(seconds=(end - start).total_seconds())}")
        self.model_args['status'] = cvx_status
        if cvx_status != 'Solved':
            utils.warning(f"MATLAB couldn't solve optimally: {cvx_status}")
        # Convert back to numpy array and ensure it's float64
        self.kernel_ = np.array(K_matlab, dtype=np.float64)
        return self

class Ineq(MVU):
    def __init__(self, model_args:dict, n_neighbors:int, eps:float=1e-3):
        super().__init__(model_args, n_neighbors, eps)
        self._mode = 1

class Nystrom(models.extensions.Nystrom, MVU):
    def __init__(self, model_args:dict, n_neighbors:int, eps:float=1e-3, ratio:int=None, subset_indices:list=None):
        MVU.__init__(self, model_args, n_neighbors, eps)
        super().__init__(ratio=ratio, subset_indices=subset_indices)

class ENG(models.extensions.ENG, MVU):
    pass

class Adaptative(models.extensions.Adaptative, MVU):
    def __init__(self, model_args:dict, n_neighbors:int, eps:float=1e-3, k_max:int=10, eta:float=0.95):
        MVU.__init__(self, model_args, n_neighbors, eps)
        super().__init__(k_max, eta)

class Adaptative2(models.extensions.Adaptative2, MVU):
    def __init__(self, model_args:dict, n_neighbors:int, eps:float=1e-3, k_max:int=10, eta:float=0.95):
        MVU.__init__(self, model_args, n_neighbors, eps)
        super().__init__(k_max, eta)

class Our(models.extensions.Our, MVU):
    def __init__(self, model_args:dict, n_neighbors:int, n_components:int):
        MVU.__init__(self, model_args, n_neighbors, n_components)
        super().__init__(bidirectional=True)



