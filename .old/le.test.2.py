
from sklearn.metrics import pairwise_distances
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
import networkx as nx

class LE:
    
    def __init__(self, X:np.ndarray, dim:int, k:int = 2, eps = None, graph:str = 'k-nearest', weights:str = 'heat kernel', 
                 sigma:float = 0.1, laplacian:str = 'unnormalized', opt_eps_jumps:float = 1.5):
        """
        LE object
        Parameters
        ----------
        
        X: nxd matrix
        
        dim: number of coordinates
        
        k: number of neighbours. Only used if graph = 'k-nearest'
        
        eps: epsilon hyperparameter. Only used if graph = 'eps'. 
        If is set to None, then epsilon is computed to be the 
        minimum one which guarantees G to be connected
        
        graph: if set to 'k-nearest', two points are neighbours 
        if one is the k nearest point of the other. 
        If set to 'eps', two points are neighbours if their 
        distance is less than epsilon
        
        weights: if set to 'heat kernel', the similarity between 
        two points is computed using the heat kernel approach.
        If set to 'simple', the weight between two points is 1
        if they are connected and 0 otherwise. If set to 'rbf'
        the similarity between two points is computed using the 
        gaussian kernel approach.
        
        sigma: coefficient for gaussian kernel or heat kernel
        
        laplacian: if set to 'unnormalized', eigenvectors are 
        obtained by solving the generalized eigenvalue problem 
        Ly = λDy where L is the unnormalized laplacian matrix.
        If set to 'random', eigenvectors are obtained by decomposing
        the Random Walk Normalized Laplacian matrix. If set to 
        'symmetrized', eigenvectors are obtained by decomposing
        the Symmetrized Normalized Laplacian
        
        opt_eps_jumps: increasing factor for epsilon
        """
        
        self.X = X
        self.dim = dim
        self.k = k
        self.eps = eps
        if graph not in ['k-nearest', 'eps']:
            raise ValueError("graph is expected to be a graph name; 'eps' or 'k-nearest', got {} instead".format(graph))
        self.graph = graph
        if weights not in ['simple', 'heat kernel', 'rbf']:
            raise ValueError("weights is expected to be a weight name; 'simple' or 'heat kernel', got {} instead".format(weights))
        self.weights = weights
        self.sigma = sigma
        self.n = self.X.shape[0]
        if laplacian not in ['unnormalized', 'random', 'symmetrized']:
            raise ValueError("laplacian is expected to be a laplacian name; 'unnormalized', 'random' or 'symmetrized', got {} instead".format(laplacian))
        self.laplacian = laplacian
        self.opt_eps_jumps = opt_eps_jumps
        if self.eps is None and self.graph == 'eps':
            self.__optimum_epsilon()
    
    def __optimum_epsilon(self):
        """
        Compute epsilon
        
        To chose the minimum epsilon which guarantees G to be 
        connected, first, epsilon is set to be equal to the distance 
        from observation i = 0 to its nearest neighbour. Then
        we check if the Graph is connected, if it's not, epsilon
        is increased and the process is repeated until the Graph
        is connected
        """
        dist_matrix = pairwise_distances(self.X)
        self.eps = min(dist_matrix[0,1:])
        con = False
        while not con:
            self.eps = self.opt_eps_jumps * self.eps
            self.__construct_nearest_graph()
            con = self.cc == 1
            print('[INFO] Epsilon: {}'.format(self.eps))
        self.eps = np.round(self.eps, 3)
    
    def __heat_kernel(self, dist):
        """
        k(x, y) = exp(- ||x-y|| / sigma )
        """
        return np.exp(- (dist*dist)/self.sigma)
    
    def __rbf(self, dist):
        """
        k(x, y) = exp(- (1/2*sigma^2) * ||x-y||^2)
        """
        return np.exp(- dist**2/ (2* (self.sigma**2) ) )
    
    def __simple(self, *args):
        return 1
    
    def __construct_nearest_graph(self):
        """
        Compute weighted graph G
        """
        similarities_dic = {'heat kernel': self.__heat_kernel,
                            'simple':self.__simple,
                            'rbf':self.__rbf}
        
        dist_matrix = pairwise_distances(self.X)
        if self.graph == 'k-nearest':
            nn_matrix = np.argsort(dist_matrix, axis = 1)[:, 1 : self.k + 1]
        elif self.graph == 'eps':
            nn_matrix = np.array([ [index for index, d in enumerate(dist_matrix[i,:]) if d < self.eps and index != i] for i in range(self.n) ])
        # Weight matrix
        
        print(nn_matrix[0])
        breakpoint()

        self._W = []
        for i in range(self.n):
            w_aux = np.zeros((1, self.n))
            similarities = np.array([ similarities_dic[self.weights](dist_matrix[i,v]) for v in nn_matrix[i]] )
            # print(similarities)
            # breakpoint()
            np.put(w_aux, nn_matrix[i], similarities)
            self._W.append(w_aux[0])
        
        self._W = np.array(self._W)


        print(self._W[0][np.where(self._W[0] != 0)])

        # D matrix
        self._D = np.diag(self._W.sum(axis=1))

        # print(self._D)

        # Check for connectivity
        self._G = self._W.copy() # Adjacency matrix
        self._G[self._G > 0] = 1
        G = nx.convert_matrix.from_numpy_array(self._G)
        self.cc = nx.number_connected_components(G) # Multiplicity of lambda = 0
        if self.cc != 1:
            warnings.warn("Graph is not fully connected, Laplacian Eigenmaps may not work as expected")
            
    def __compute_unnormalized_laplacian(self):
        self.__construct_nearest_graph()
        self._L = self._D - self._W
        return self._L
    
    def __compute_normalized_random_laplacian(self):
        self.__construct_nearest_graph()
        self._Lr = np.eye(*self._W.shape) - (np.diag(1/self._D.diagonal())@self._W)
        return self._Lr
    
    def __compute_normalized_symmetrized_laplacian(self):
        self.__construct_nearest_graph()
        self.__compute_unnormalized_laplacian()
        d_tilde = np.diag(1/np.sqrt(self._D.diagonal()))
        self._Ls = d_tilde @ ( self._L @ d_tilde )
        return self._Ls
    
    def transform(self):
        """
        Compute embedding
        """
        
        m_options = {
            'unnormalized':self.__compute_unnormalized_laplacian,
            'random':self.__compute_normalized_random_laplacian,
            'symmetrized':self.__compute_normalized_symmetrized_laplacian
        }
        
        L = m_options[self.laplacian]()
        # print(L, self._D)
        
        if self.laplacian == 'unnormalized':
            eigval, eigvec = eigh(L, self._D) # Generalized eigenvalue problem
        else:
            eigval, eigvec = np.linalg.eig(L)
            
        order = np.argsort(eigval)
        self.Y = eigvec[:, order[self.cc:self.cc+self.dim + 1]]
            
        return self.Y
    
    def plot_embedding_2d(self, colors, grid = True, dim_1 = 1, dim_2 = 2, cmap = None, size = (15, 10)):
        if self.dim < 2 and dim_2 <= self.dim and dim_1 <= self.dim:
            raise ValueError("There's not enough coordinates")
        
        # plt.style.use('seaborn-whitegrid')
        fig = plt.figure(figsize=size)
        plt.axhline(c = 'black', alpha = 0.2)
        plt.axvline(c = 'black', alpha = 0.2)
        if cmap is None:
            plt.scatter(self.Y[:, dim_1 - 1], self.Y[:, dim_2 - 1], c = colors)
            
        plt.scatter(self.Y[:, dim_1 - 1], self.Y[:, dim_2 - 1], c = colors, cmap=cmap)
        plt.grid(grid)
        if self.graph == 'k-nearest':
            title = 'LE with k = {} and weights = {}'.format(self.k, self.weights)
        else:
            title = 'LE with $\epsilon$ = {} and weights = {}'.format(self.eps, self.weights)
        plt.title(title)
        plt.xlabel('Coordinate {}'.format(dim_1))
        plt.ylabel('Coordinate {}'.format(dim_2))
        plt.show()
    
    def plot_embedding_3d(self, colors, grid = True, dim_1 = 1, dim_2 = 2, dim_3 = 3, cmap = None, size = (15, 10)):
        if self.dim < 3 and dim_2 <= self.dim and dim_1 <= self.dim and dim_3 <= self.dim:
            raise ValueError("There's not enough coordinates")
        
        plt.style.use('seaborn-whitegrid')
        fig = plt.figure(figsize=size)
        ax = fig.add_subplot(111, projection="3d")
        if cmap is None:
            ax.scatter(self.Y[:, dim_1 - 1], self.Y[:, dim_2 - 1], self.Y[:, dim_3 - 1], c = colors)
        ax.scatter(self.Y[:, dim_1 - 1], self.Y[:, dim_2 - 1], self.Y[:, dim_3 - 1], c = colors, cmap = cmap)
        plt.grid(grid)
        ax.axis('on')
        if self.graph == 'k-nearest':
            title = 'LE with k = {} and weights = {}'.format(self.k, self.weights)
        else:
            title = 'LE with $\epsilon$ = {} and weights = {}'.format(self.eps, self.weights)
        plt.title(title)
        ax.set_xlabel('Coordinate {}'.format(dim_1))
        ax.set_ylabel('Coordinate {}'.format(dim_2))
        ax.set_zlabel('Coordinate {}'.format(dim_3))
        plt.show()


# import numpy as np
# import scipy.sparse as sp
# import scipy.sparse.linalg as spla
# import networkx as nx
# from sklearn.metrics.pairwise import euclidean_distances

# def laplacian_eigen(X, no_dims=2, k=12, sigma=1.0):
#     """
#     Performs non-linear dimensionality reduction using Laplacian Eigenmaps.

#     Parameters:
#     X       : (n_samples, n_features) data matrix
#     no_dims : (int) target dimension (default=2)
#     k       : (int) number of neighbors (default=12)
#     sigma   : (float) heat kernel parameter (default=1)

#     Returns:
#     mappedX : (n_samples, no_dims) reduced dimensional data
#     mapping : (dict) dictionary containing transformation details
#     """
    
#     # Construct neighborhood graph
#     print("Constructing neighborhood graph...")
#     dist_matrix = euclidean_distances(X, X)
    
#     # Sort distances and keep only k nearest neighbors
#     sorted_indices = np.argsort(dist_matrix, axis=1)
#     # print(sorted_indices[:, 1:k+1][0])
#     # breakpoint()
    
    
#     for i in range(dist_matrix.shape[0]):
#         dist_matrix[i, sorted_indices[i, k+1:]] = 0  # Keep k neighbors only
    
#     # print(dist_matrix[0], np.count_nonzero(dist_matrix[0]))
#     # breakpoint()

#     # Symmetrize distance matrix
#     G = np.maximum(dist_matrix, dist_matrix.T)
#     G = np.square(G)
    
#     mapping = {}
#     mapping['max_dist'] = np.max(G)
#     G /= mapping['max_dist']  # Normalize distances
    
#     # Only embed the largest connected component
#     graph = nx.from_numpy_array(G)
#     largest_cc = max(nx.connected_components(graph), key=len)
#     conn_comp = np.array(list(largest_cc))
#     G = G[np.ix_(conn_comp, conn_comp)]
    
#     # Compute weight matrix using Gaussian kernel
#     print("Computing weight matrices...")
#     W = np.exp(-G / (2 * sigma**2)) * (G > 0)
    
#     # Compute diagonal degree matrix
#     D = np.diag(W.sum(axis=1))

#     print(D)
    
#     # Compute Laplacian matrix
#     L = D - W

#     # print(D)


#     d_tilde = np.diag(1/np.sqrt(D.diagonal()))
#     L = d_tilde @ ( L @ d_tilde )

    
#     # Solve generalized eigenproblem Lx = λDx
#     print("Constructing Eigenmaps...")
#     L = sp.csc_matrix(L)
#     D = sp.csc_matrix(D)
    
#     eigenvalues, eigenvectors = spla.eigsh(L, k=no_dims + 1, M=D, which='SM')

#     # Sort eigenvectors in ascending order
#     sorted_indices = np.argsort(eigenvalues)
#     eigenvalues = eigenvalues[sorted_indices][1:no_dims + 1]
#     mappedX = eigenvectors[:, sorted_indices[1:no_dims + 1]]
    
#     # Store mapping details
#     mapping['K'] = G
#     mapping['vec'] = mappedX
#     mapping['val'] = eigenvalues
#     mapping['X'] = X[conn_comp]
#     mapping['sigma'] = sigma
#     mapping['k'] = k
#     mapping['conn_comp'] = conn_comp

#     return mappedX, mapping


# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import make_swiss_roll

# # Generate Swiss Roll dataset
# n_samples = 1000
# X, color = make_swiss_roll(n_samples, noise=0.1)

# # Apply Laplacian Eigenmaps
# mappedX, mapping = laplacian_eigen(X, no_dims=2, k=10, sigma=5.0)

# # Plot original 3D Swiss Roll
# fig = plt.figure(figsize=(12, 5))

# ax1 = fig.add_subplot(121, projection='3d')
# ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap='Spectral')
# ax1.set_title("Original Swiss Roll")

# # Plot 2D embedding
# ax2 = fig.add_subplot(122)
# ax2.scatter(mappedX[:, 0], mappedX[:, 1], c=color, cmap='Spectral')
# ax2.set_title("Laplacian Eigenmaps Embedding")

# plt.show()




# # from le import LE
# le = LE(X, dim = 3, k=10, graph = 'k-nearest', weights = 'heat kernel', 
#         sigma = 5, laplacian = 'symmetrized')
# # le = LE(X, dim = 3, eps = 1.97, graph = 'eps', weights = 'heat kernel', 
# #         sigma = 5, laplacian = 'symmetrized')
# Y_t = le.transform()

# from plot import plot
# plot(Y_t)

# le.plot_embedding_2d(color, cmap=plt.cm.jet, grid = False, size = (14, 6))

