
import numpy as np
import cvxpy as cvx

import mosek
from scipy.sparse import csgraph, csr_matrix
from scipy.spatial.distance import cdist

import models.neighbourhood
import models.extensions

class MVU(models.Neighbourhood):

    def __init__(self, model_args:dict, n_neighbors:int, eps:float=1e-3):
        super().__init__(model_args, n_neighbors)
        self.eps = eps
        self._mode = 0

    def _neigh_matrix(self, X:np.ndarray):
        return self.k_neigh(X, bidirectional=True, common_neighbors=False)[1]

    def _fit(self, X:np.ndarray, starting_K:np.ndarray=None):
        """Fit the MVU model and compute the low-dimensional embeddings."""
        n_samples = X.shape[0]


        # cc, labels = csgraph.connected_components(self.NM, directed=False)
        # if cc > 1:
        #     print(f"Warning: {cc} components found. Using largest component and nystrom approximating the rest.")
        #     largest_component_indexes = np.where(labels == np.argmax(np.bincount(labels)))[0]
        #     model = Nystrom(self.model_args, self.n_neighbors, self.eps, subset_indices=largest_component_indexes)
        #     model.neigh_matrix(X[largest_component_indexes])
        #     model._fit(X)
        #     self.kernel_ = model.kernel_
        #     return self

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
        inner_prod = (inner_prod / ratio).astype(inner_prod.dtype)
        # print(inner_prod.dtype)

        K = cvx.Variable((n_samples, n_samples), PSD=True)
        if starting_K is not None:
            K.value = starting_K #TODO: previously optimized K's are not fully PSD (check the smallest eigenvalues, maybe apply same clause as in the embedding (if neg, 0))
        else:
            # K.value = np.zeros((n_samples, n_samples))
            K.value = inner_prod
        _X = cvx.Constant(inner_prod)

        objective = cvx.Maximize(cvx.trace(K))
        constraints = [cvx.sum(K) == 0]

        # add distance-preserving constraints
        for i in range(n_samples):
            for j in range(n_samples):
                if self.NM[i, j] != 0:
                    if self._mode == 0:
                        constraints.append(
                            (_X[i,i] - 2 * _X[i,j] + _X[j,j]) - 
                            (K[i, i] - 2 * K[i, j] + K[j, j]) == 0
                        )
                    elif self._mode == 1:
                        constraints.append(
                            (_X[i,i] - 2 * _X[i,j] + _X[j,j]) - 
                            (K[i, i] - 2 * K[i, j] + K[j, j]) >= 0
                        )

        print("Length of constraints:",len(constraints))

        problem = cvx.Problem(objective, constraints)
        problem.solve(solver="SCS", verbose=True, eps=self.eps)
        if problem.status != cvx.OPTIMAL:
            print("Warning: Problem not solved optimally. Status:", problem.status)
            raise ValueError("Problem not solved optimally")
        print(problem.status) # TODO: when threaded, from here on nothing is printed, probably gets silently killed; when single-threaded, it gets killed (oom, MOSEK)
        self.kernel_ = K.value
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



