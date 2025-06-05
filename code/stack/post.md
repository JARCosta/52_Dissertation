I am having a problem with the implementation of Maximum Variance Unfolding (MVU), a non-linear dimensionality reduction method, using CVX in MATLAB, which is called from a Python wrapper.

My main question is: **Is MOSEK's `UNKOWN` (`Inaccurate/Unbounded`) status an actual formulation problem or is it just a matter of tolerances?**

The problem is found right in the implementation of the usual optimisation problem of MVU. When letting CVX pick the solver it uses SDPT3, which seems to be having a problem in the middle of the optimisation process and stops half way, returning:
```
 5|1.000|0.093|8.4e-09|9.8e-01|1.2e+07|-1.005117e+09 -1.986820e+01| 0:0:08| chol  2  2 
 6|1.000|0.011|3.2e-04|9.7e-01|8.7e+09|-6.902778e+12 -2.091421e+00| 0:0:09| chol  2  2 
  stop: primal infeas has deteriorated too much, 4.1e+01
 7|1.000|0.001|3.2e-04|9.7e-01|8.7e+09|-6.902778e+12 -2.091421e+00| 0:0:11|
  prim_inf,dual_inf,relgap = 3.15e-04, 9.71e-01, 1.27e-03
  sqlp stop: dual problem is suspected of being infeasible
-------------------------------------------------------------------
 number of iterations   =  7
 residual of dual infeasibility
 certificate X          = 3.28e-11
 reldist to infeas.    <= 2.73e-13
 Total CPU time (secs)  = 10.77  
 CPU time per iteration = 1.54
 termination code       =  2
 DIMACS: 2.8e-03  0.0e+00  1.6e+01  0.0e+00  -1.0e+00  1.3e-03
-------------------------------------------------------------------
 
------------------------------------------------------------
Status: Unbounded
Optimal value (cvx_optval): +Inf
```

When setting the solver to MOSEK, apparently, it computes the problem correctly.
The problem with MOSEK is that it returns `Problem status: UNKNOWN` and `status: Innacurate/Unbounded`.
I've tried the `MSK_DPAR_INTPNT_CO_TOL_PFEAS`, `MSK_DPAR_INTPNT_CO_TOL_DFEAS` solver settings, but I'm not sure if I'm using them correctly, or if there are better options.


```
ITE PFEAS    DFEAS    GFEAS    PRSTATUS   POBJ              DOBJ              MU       TIME
38  3.4e-08  1.4e-05  3.6e-09  9.99e-01   -7.951611627e+05  -7.951611461e+05  2.8e-11  220.41
39  3.4e-08  1.4e-05  3.6e-09  9.99e-01   -7.951611627e+05  -7.951611461e+05  2.8e-11  237.47
40  3.4e-08  1.4e-05  3.6e-09  1.00e+00   -7.951611627e+05  -7.951611461e+05  2.8e-11  255.11
Optimizer terminated. Time: 275.84

Interior-point solution summary
  Problem status  : UNKNOWN
  Solution status : UNKNOWN
  Primal.  obj: -7.9516116275e+05   nrm: 3e+03    Viol.  con: 9e-04    barvar: 0e+00
  Dual.    obj: -7.9516114605e+05   nrm: 3e+06    Viol.  con: 0e+00    barvar: 4e-01
------------------------------------------------------------
Status: Inaccurate/Unbounded
Optimal value (cvx_optval): +Inf
```

Although I was able to make it return `Status: Solved` when reducing the feasibility tolerance, it was only possible in some of my benchmark datasets: the swissroll dataset (1e-7) and twinpeaks dataset (1e-5). The helix dataset is particularly stubborn, since not even 1e-1 was enough to return a `Status: Solved`.

While [Mosek's documentation](https://docs.mosek.com/11.0/dotnetapi/debugging-log.html#continuous-problem) says the solution might be useful despite the solver returning `Status: Inaccurate/Unbounded`, why couldn't SDPT3 or other solvers find a solution?

Appreciate any help.
Thank you in advance.


Here is an example code that I'm using:
```python
import numpy as np
import matlab.engine
import os


eng = None

os.environ['LOKY_MAX_CPU_COUNT'] = str(os.cpu_count())
os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())


############################################################
# DATASETS #################################################
############################################################

import numpy as np

def swiss(n_points, noise, random_state=None):
    np.random.seed(random_state) if random_state is not None else None

    t = (3 * np.pi / 2) * (1 + 2 * np.random.rand(n_points, 1))
    height = 30 * np.random.rand(n_points, 1)
    X = np.hstack([
        t * np.cos(t),
        height,
        t * np.sin(t)
    ]) + noise * np.random.randn(n_points, 3)

    labels = np.remainder(np.sum(np.hstack([np.round(t / 2), np.round(height / 12)]), axis=1), 2)
    t = np.hstack([t, height])

    return X, labels, t


def twinpeaks(n_points, noise, random_state=None):
    np.random.seed(random_state) if random_state is not None else None
    
    xy = 1 - 2 * np.random.rand(2, n_points)
    X = np.hstack([xy.T, (np.sin(np.pi * xy[0, :]) * np.tanh(3 * xy[1, :]))[:, None]]) + noise * np.random.randn(n_points, 3)
    X[:, 2] *= 10
    labels = np.remainder(np.sum(np.round((X - np.min(X, axis=0)) / 10), axis=1), 2)

    return X, labels, None

def helix(n_points, noise, random_state=None):
    np.random.seed(random_state) if random_state is not None else None

    t = np.linspace(1, n_points, n_points)[:, None] / n_points
    t = (t ** 1.0) * 2 * np.pi
    X = np.hstack([(2 + np.cos(8 * t)) * np.cos(t), (2 + np.cos(8 * t)) * np.sin(t), np.sin(8 * t)]) + noise * np.random.randn(n_points, 3)
    labels = np.remainder(np.round(t * 1.5), 2)

    return X, labels, t

############################################################
# UTILS ####################################################
############################################################
from scipy.sparse import csr_matrix

def k_neigh(data, n_neighbors, bidirectional=False):
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
    def intra_dist_matrix(data: np.ndarray):
        from sklearn.neighbors import NearestNeighbors
        
        n_samples = data.shape[0]
        dist, indices = NearestNeighbors(n_neighbors=n_samples).fit(data).kneighbors(data)
        dist_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            dist_matrix[i][indices[i]] = dist[i]
        return dist_matrix

    k = n_neighbors
    n_samples = data.shape[0]
    dist_matrix = intra_dist_matrix(data[:, :5])
    neigh_matrix = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        nearest_indices = np.argsort(dist_matrix[i])[0:k + 1]
        
        nearest_indices = np.delete(nearest_indices, np.where(nearest_indices == i))
        nearest_indices = nearest_indices[0:k]

        for j in nearest_indices:
            neigh_matrix[i, j] = dist_matrix[i, j] if dist_matrix[i, j] > 0 else 1e-10

    if bidirectional: # remove duplicate connections
        neigh_bidirectional = np.maximum(neigh_matrix, neigh_matrix.T)
        neigh_matrix = np.triu(neigh_bidirectional)


    return neigh_matrix

def k_graph(neigh_matrix:np.ndarray):
    return csr_matrix(neigh_matrix != 0)


############################################################
# MVU ######################################################
############################################################



class MVU:

    def __init__(self, n_neighbors:int=5):
        global eng
        if eng is None:
            eng = matlab.engine.start_matlab()
            # Add the directory containing the MATLAB function to the MATLAB path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            eng.addpath(current_dir, nargout=0)
        self.n_neighbors = n_neighbors


    def _fit(self, X:np.ndarray):
        """Fit the MVU model and compute the low-dimensional embeddings using MATLAB."""
        self.NM = k_neigh(X, self.n_neighbors, bidirectional=True)
        # Convert data to MATLAB format
        X_matlab = matlab.double(X.tolist())
        
        # NM is a nxn sparse matrix of neighbourhood connections (distance(i,j) where j is in the neighbourhood of i)
        rows, cols = k_graph(self.NM).nonzero()
        
        neighbor_pairs = [[int(i) + 1, int(j) + 1] for i, j in zip(rows, cols)]

        N_matlab = matlab.double(neighbor_pairs)
        
        K_matlab, cvx_status = eng.solve_mvu_optimization(X_matlab, N_matlab, nargout=2)
        self.kernel_ = np.array(K_matlab, dtype=np.float64)
        return self


    def _transform(self):
        """Transform the data to the low-dimensional space."""
        eigenvalues, eigenvectors = np.linalg.eigh(self.kernel_)

        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:, idx]

        eigenvalues_idx = []
        for i in idx:
            if eigenvalues[i] > (np.median(eigenvalues) + np.std(eigenvalues)):
                eigenvalues_idx.append(i)
        
        eigenvalues = eigenvalues[eigenvalues_idx]
        eigenvectors = eigenvectors[:, eigenvalues_idx]

        embedding = eigenvectors @ np.diag(eigenvalues)
        return embedding


    def fit_transform(self, X:np.ndarray):
        """Fit the model and transform the data to the low-dimensional space."""
        return self._fit(X)._transform()




if __name__ == "__main__":
    # X, labels, t = swiss(1000, 0.1, 11)
    # X, labels, t = twinpeaks(1000, 0.1, 11)
    X, labels, t = helix(1000, 0.1, 11)
    print(f"Loaded data")
    mvu = MVU(n_neighbors=5)
    print("Initialized MVU")
    embedding = mvu.fit_transform(X)
    print(embedding.shape)
```
function [G, cvx_status] = solve_mvu_optimization(X, N)
    % SOLVE_MVU_OPTIMIZATION Solves the Maximum Variance Unfolding optimization problem
    % Inputs:
    % X: n-by-D data matrix (rows are points)
    % N: set of index pairs (i,j) in the neighborhood (e.g., from k-NN)

    n = size(X, 1);

    % inner_prod = X * X';
    D = pdist2(X, X).^2; % squared pairwise distances

    % ==== Step 3: Solve MVU via CVX ====
    cvx_begin sdp
        % cvx_solver sdpt3
        cvx_solver mosek

        variable G(n, n) symmetric % triang( n*n )
        maximize( trace(G) )
        subject to
            G >= 0;
            sum(G(:)) == 0;

            % Extract indices for vectorized operations
            i_indices = N(:, 1);
            j_indices = N(:, 2);
            
            G_diag = diag(G);
            gram_distances = G_diag(i_indices) + G_diag(j_indices) - 2*G(sub2ind([n,n], i_indices, j_indices));
            
            % inner_prod_diag = diag(inner_prod);
            % distances = inner_prod_diag(i_indices) + inner_prod_diag(j_indices) - 2*inner_prod(sub2ind([n,n], i_indices, j_indices));
            % gram_distances == distances;
            gram_distances == D(sub2ind([n,n], i_indices, j_indices));
        
            cvx_end
    end