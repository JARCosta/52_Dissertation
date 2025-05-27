import numpy as np
import cvxpy as cvx

import mosek
from scipy.sparse import csgraph, csr_matrix
from scipy.spatial.distance import cdist
import matlab.engine
import os

import utils
import models.neighbourhood
import models.extensions

eng = None

class MVU(models.Neighbourhood):

    def __init__(self, model_args:dict, n_neighbors:int, eps:float=1e-3):
        global eng
        if eng is None:
            eng = matlab.engine.start_matlab()
            # Add the directory containing the MATLAB function to the MATLAB path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            eng.addpath(current_dir, nargout=0)
        super().__init__(model_args, n_neighbors)
        self.eps = eps
        self._mode = 0

    def _neigh_matrix(self, X:np.ndarray):
        return self.k_neigh(X, bidirectional=True, common_neighbors=False)[1]

    def _fit(self, X:np.ndarray, starting_K:np.ndarray=None):
        """Fit the MVU model and compute the low-dimensional embeddings using MATLAB."""
        n_samples = X.shape[0]

        cc, labels = csgraph.connected_components(self.NM, directed=False)
        if cc > 1:
            print(f"Warning: {cc} components found. Adding shortest connections possible to merge components.")
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

        # inner product matrix of the original data
        inner_prod = (X @ X.T)
        ratio = 10**(np.round(np.log10(np.max(inner_prod)))-2)
        inner_prod = (inner_prod / ratio).astype(np.float64)

        # Convert data to MATLAB format
        X_matlab = matlab.double(X.tolist())
        inner_prod_matlab = matlab.double(inner_prod.tolist())
        NM_matlab = matlab.double(self.NM.tolist())
        
        utils.stamp.print(f"*\t {self.model_args['model']}\t calling MATLAB")
        K_matlab, cvx_status = eng.solve_mvu_optimization(X_matlab, inner_prod_matlab, NM_matlab, 
                                            float(self.eps), self._mode, nargout=2)
        if cvx_status != 'Solved':
            utils.warning(f"MATLAB couldn't solve optimally: {cvx_status}")
        # Convert back to numpy array and ensure it's float64
        self.kernel_ = np.array(K_matlab, dtype=np.float64)
        # Scale back the kernel matrix
        self.kernel_ = self.kernel_ * ratio
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



