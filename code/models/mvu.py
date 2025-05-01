
import numpy as np
import cvxpy as cvx

import mosek
from scipy.sparse import csgraph, csr_matrix

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


        cc, labels = csgraph.connected_components(self.NM, directed=False)
        if cc > 1:
            print(f"Warning: {cc} components found. Using largest component and nystrom approximating the rest.")
            largest_component_indexes = np.where(labels == np.argmax(np.bincount(labels)))[0]
            model = Nystrom(self.model_args, self.n_neighbors, self.eps, subset_indices=largest_component_indexes)
            model.neigh_matrix(X[largest_component_indexes])
            model._fit(X)
            self.kernel_ = model.kernel_
            return self

        # inner product matrix of the original data
        inner_prod = (X @ X.T)
        ratio = 10**(np.round(np.log10(np.max(inner_prod)))-2)
        inner_prod = inner_prod / ratio

        # inner_prod = inner_prod * (100 / np.median(inner_prod)) #np.max(inner_prod))
        if starting_K is None:
            starting_K = inner_prod

        K = cvx.Variable((n_samples, n_samples), PSD=True)
        K.value = starting_K #TODO: previously optimized K's are not fully PSD (check the smallest eigenvalues, maybe apply same clause as in the embedding (if neg, 0))
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
        problem.solve(solver="MOSEK", verbose=True, eps=self.eps)
        if problem.status != cvx.OPTIMAL:
            print("Warning: Problem not solved optimally. Status:", problem.status)
            raise ValueError("Problem not solved optimally")
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



