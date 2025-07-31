import numpy as np

from scipy.sparse import csgraph
from scipy.spatial.distance import cdist
import matlab.engine
import os

import plot
import utils
import models

eng = None

class MVU(models.Neighbourhood):

    def __init__(self, model_args:dict, n_neighbors:int):
        super().__init__(model_args, n_neighbors)
        self._mode = 0
        
        global eng
        if eng is None:
            eng = matlab.engine.start_matlab()
            # Add the directory containing the MATLAB function to the MATLAB path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            eng.addpath(current_dir, nargout=0)
            utils.stamp.print(f"*\t {self.model_args['model']}\t MATLAB started")

    def _neigh_matrix(self, X:np.ndarray):
        return utils.neigh_matrix(X, self.n_neighbors, bidirectional=True)

    def _fit(self, X:np.ndarray):
        """Fit the MVU model and compute the low-dimensional embeddings using MATLAB."""
        
        self.X = X

        cc, labels = csgraph.connected_components(self.NM, directed=False)
        if cc > 1:
            oldNM = self.NM.copy()

            # component_idx:list[int] = [np.where(labels == c)[0] for c in range(cc)]
            # self.component_idx = component_idx

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
                plot.plot_two(X, X, oldNM, self.NM, scale=False, scale2=False, block=False, title=f"{self.model_args['dataname']} {self.model_args['#neighs']} neighbors")

        
        # Extract index pairs from sparse matrix
        # neigh_graph.nonzero() returns (row_indices, col_indices) of non-zero elements
        np.set_printoptions(threshold=200)
        
        
        if self.model_args['verbose']:
            print()
            print(f"Original distances:")
            print(f"\t min: {np.min(self.NM[self.NM > 0]):.4f}, max: {np.max(self.NM):.4f}")
        
        self.ratio = None
        if hasattr(self, 'ratio') and self.ratio is None:
            scaling_ratio = round(np.log10(np.max(self.NM))) - 2
            scaling_ratio = 10**(-scaling_ratio)

            self.NM = self.NM * scaling_ratio
            self.recovering_ratio = 1/scaling_ratio

            if self.model_args['verbose']:
                print(f"scalling ratio: {scaling_ratio}")
                print(f"After scalling:")
                print(f"\t min: {np.min(self.NM[self.NM > 0]):.4f}, max: {np.max(self.NM):.4f}")
                print()
                
                temp = np.where(self.NM > 0)
                print("NM input:")
                for i in range(-5, 5):
                    print(f"{temp[0][i]} -> {temp[1][i]} = {self.NM[temp[0][i], temp[1][i]]:.4f}")
                print()

        self.NM[self.NM > 0] = self.NM[self.NM > 0]**2
        NM_matlab = matlab.double(self.NM.tolist())
        n_matlab = matlab.double(X.shape[0])
        
        utils.stamp.print(f"*\t {self.model_args['model']}\t calling MATLAB")
        
        if not hasattr(self, 'eps') or self.eps is None:
            self.eps = 1e-8
        K_matlab, cvx_status = eng.solve_mvu_optimization(n_matlab, NM_matlab, self.eps, nargout=2)
        utils.stamp.print(f"*\t CVX\t {cvx_status}\t trace: {np.trace(K_matlab):.2f}\t precision:{self.eps}")
        self.eps = None

        self.model_args['status'] = cvx_status
        if cvx_status != 'Solved':
            utils.warning(f"MATLAB couldn't solve optimally: {cvx_status}")
        # Convert back to numpy array and ensure it's float64
        self.kernel_ = np.array(K_matlab, dtype=np.float64)
        return self

    def _transform(self):
        embedding = super()._transform()

        if 'status' in self.model_args and self.model_args['status'] != 'Solved':
            utils.hard_warning(f"Spectral embedding failed: {self.model_args['status']}. Scaling output.")
            Yi_NM = utils.neigh_matrix(embedding, self.n_neighbors)
            min_Yi, max_Yi = np.min(Yi_NM[Yi_NM != 0]), np.max(Yi_NM)


            Xi_NM = utils.neigh_matrix(self.X, self.n_neighbors)
            min_Xi, max_Xi = np.min(Xi_NM[Xi_NM != 0]), np.max(Xi_NM)
            # recovering_ratio = np.mean([min_Xi/min_Yi, max_Xi/max_Yi])
            recovering_ratio = 10**(round(np.log10(max_Xi/max_Yi)))
            
            embedding = embedding * recovering_ratio

            if self.model_args['verbose']:
                print()
                print(f"original\n\t min: {min_Xi:.4f}, max: {max_Xi:.4f}")
                print(f"before recovery scalling\n\t min: {np.min(Yi_NM[Yi_NM != 0]):.4f}, max: {np.max(Yi_NM):.4f}")
                print(f"recovery ratio: {recovering_ratio:.4f}")
                Yi_NM = utils.neigh_matrix(embedding, self.n_neighbors)
                print(f"after recovery scalling\n\t min: {np.min(Yi_NM[Yi_NM != 0]):.4f}, max: {np.max(Yi_NM):.4f}")
                print()
        
        return embedding


class Ineq(MVU):
    def __init__(self, model_args:dict, n_neighbors:int):
        super().__init__(model_args, n_neighbors)
        self._mode = 1

class Nystrom(models.extensions.Nystrom, MVU):
    def __init__(self, model_args:dict, n_neighbors:int, ratio:int=None, subset_indices:list=None):
        MVU.__init__(self, model_args, n_neighbors)
        super().__init__(ratio=ratio, subset_indices=subset_indices)

class ENG(models.extensions.ENG, MVU):
    pass

class Adaptative(models.extensions.Adaptative, MVU):
    def __init__(self, model_args:dict, n_neighbors:int, k_max:int=10, eta:float=0.95):
        MVU.__init__(self, model_args, n_neighbors)
        super().__init__(k_max, eta)

class Adaptative2(models.extensions.Adaptative2, MVU):
    def __init__(self, model_args:dict, n_neighbors:int, k_max:int=10, eta:float=0.95):
        MVU.__init__(self, model_args, n_neighbors)
        super().__init__(k_max, eta)
    
class Our(models.extensions.Our, MVU):
    def __init__(self, model_args:dict, n_neighbors:int):
        MVU.__init__(self, model_args, n_neighbors)
        super().__init__(bidirectional=True)

class Based(models.extensions.Based, MVU):
    def __init__(self, model_args:dict, k1:int, k2:int):
        MVU.__init__(self, model_args, n_neighbors=k1)
        super().__init__(k1, k2)

