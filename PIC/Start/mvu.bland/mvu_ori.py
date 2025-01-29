import cvxpy as cvx
import numpy as np
from sklearn.neighbors import NearestNeighbors
from itertools import combinations
from sklearn.preprocessing import minmax_scale
import time


def _is_graph_connected(graph_matrix):
	"""
	Determines whether a given (undirected) graph is connected.
	
	:param graph_matrix: a NumPy (N, N) binary ndarray representing a neighborhood graph.
	:return: True if the graph is connected, False otherwise.
	"""
	def get_first_node(nodes, visited):
		while True:
			node = nodes.pop()

			if node not in visited:
				return node

		return -1

	def dfs(graph_matrix, nodes_left, visited):
		to_visit = set()
		first_node = get_first_node(nodes_left, visited)
		component_part = set()
		
		to_visit.add(first_node)
		component_part.add(first_node)

		while to_visit:
			current = to_visit.pop()

			visited.add(current)

			current_row_adjacencies = graph_matrix[current, :]
			current_row_neighbors = np.where(current_row_adjacencies == 1)[0]

			# add neighbors of current node to visit in the future,
			# if they haven't been visited already
			neighbors_to_visit = set([neighbor for neighbor in current_row_neighbors
						if neighbor not in visited # COMPONENT_PART INSTEAD OF VISITED?
						and neighbor not in to_visit])
			to_visit |= neighbors_to_visit
			component_part |= neighbors_to_visit

		return component_part

	def components_connected(graph_matrix, c1, c2):
		for node in c1:
			node_neighbors = np.where(graph_matrix[node, :] == 1)[0]

			for neighbor in node_neighbors:
				if neighbor in c2:
					return True

		for node in c2:
			node_neighbors = np.where(graph_matrix[node, :] == 1)[0]

			for neighbor in node_neighbors:
				if neighbor in c1:
					return True

		return False

	n_nodes = graph_matrix.shape[0]
	visited = set()
	nodes_left = set(range(n_nodes))
	component_parts = []

	while len(nodes_left) != 0:
		component_parts.append(dfs(graph_matrix, nodes_left, visited))
		nodes_left -= visited

	while len(component_parts) > 1:
		joined = False

		for c1_idx, c2_idx in combinations(range(len(component_parts)), 2):
			c1 = component_parts[c1_idx]
			c2 = component_parts[c2_idx]
			if components_connected(graph_matrix, c1, c2):
				joined_component = c1.union(c2)
				del component_parts[c2_idx]
				del component_parts[c1_idx]	# order matters!
				component_parts.append(joined_component)
				joined = True
				break

		if not joined:
			return False

	return True


def MVU(data, n_neighbors=4, rescale_factor=1e-6, solution_tolerance=1e-2, save_filename=None, nn_file=None, start=None, stop=None):
	"""
	Performs maximum variance unfolding on the given data.
	:param data: a NumPy (n_points, n_features) array with the data.
	:param n_neighbors: number of neighbors of each point to preserve distances to.
	:param rescale_factor: factor by which to multiply the inner product matrix
		of the original data. Use to help with numerical stability.
	:solution_tolerance: solution tolerance for the CVX solver.
	:return: the solution, i.e., an inner product matrix of the embedded data.
	"""
	# number of data points
	n_points = data.shape[0]
	# create sparse matrix (n_points x n_points) with the connections;
	# each point is its own closest neighbor, use +1 to account for that
	nn = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(data)
	nn = nn.kneighbors_graph(data).todense()
	nn = np.array(nn)
	print("Created NN matrix!")

	for i in range(n_points):
		# check that each node has exactly n_neighbors
		if np.count_nonzero(nn[i, :]) != n_neighbors + 1:
			print(f"ERROR: data point {i} only has {np.count_nonzero(nn[i, :])} \
					neighbors; should have {n_neighbors}.")
			return None, None
		# remove all connections of each node to itself
		nn[i, i] = 0

	if nn_file:
		full_nn = np.load(nn_file)
		print("Actually new graph!")
		n_total_points = full_nn.shape[0]

		nn = full_nn[start:stop, start:stop]

	# graph must be connected for MVU to work
	if not _is_graph_connected(nn):
		print(f"ERROR: graph of shape {nn.shape} is not connected for {n_neighbors} neighbors")
		return None, None

	print("Graph is connected!")

	# inner product matrix of the original data; rescale for numerical stability
	X = cvx.Constant(data.dot(data.T) * rescale_factor)
	# inner product matrix of the projected points; PSD constrains to be both PSD and symmetric
	G = cvx.Variable((n_points, n_points), PSD=True)
	G.value = np.zeros((n_points, n_points))

	# spread out points in target manifold
	objective = cvx.Maximize(cvx.trace(G))
	# G must be zero-centered
	constraints = [cvx.sum(G) == 0]

	# add distance-preserving constraints
	for i in range(n_points):
		for j in range(n_points):
			if nn[i, j] == 1:
				constraints.append(
					(X[i, i] - 2 * X[i, j] + X[j, j]) -
					(G[i, i] - 2 * G[i, j] + G[j, j]) == 0
				)

	problem = cvx.Problem(objective, constraints)
	print("Gonna solve " + str(n_neighbors) + " at " + str(time.ctime()), flush=True)
	problem.solve(solver="SCS", eps=solution_tolerance, verbose=True)
	print("Solved nn " + str(n_neighbors) + " at " + str(time.ctime()), flush=True)

	if G.value is not None and save_filename is not None:
		np.save(save_filename, G.value)

	return G.value, problem.status


def MVU_embedded_data(inner_product_matrix, n_top_eigenvalues=8):
	# eigendecompose the inner product matrix
	eigenvalues, eigenvectors = np.linalg.eig(inner_product_matrix)

	# take the indices of the greatest eigenvalues
	# note that the eigenvalues are in a vector (n, 1)
	top_eigenvalue_indices = eigenvalues.argsort()[::-1][:n_top_eigenvalues]

	# get top eigenvalues (n,) and eigenvectors (n, n) - with n small
	top_eigenvalues = eigenvalues[top_eigenvalue_indices]
	top_eigenvectors = eigenvectors[:, top_eigenvalue_indices]

	# diagonalize top eigenvalue vector to (n, n) matrix
	top_eigenvalue_diag = np.diag(top_eigenvalues ** 0.5)
	# compute embedded data = Q * lambda^(1/2)
	embedded_data = np.dot(top_eigenvectors, top_eigenvalue_diag)

	return embedded_data, top_eigenvalues


def load_teapots():
	from scipy.io import loadmat
	data = loadmat('datasets/teapots.mat')
	data = data["Input"][0][0][0]

	# make it (400 x 23028)
	return data.T.astype(np.float64)


def load_faces():
	from scipy.io import loadmat
	data = loadmat('datasets/faces.mat')

	data = data["ff"]

	# make it (1965 x 560)
	return data.T.astype(np.float64)

if __name__ == '__main__':
	dataset = "teapots"
	# data = np.load(f'datasets/{dataset}.npy')
	data = load_teapots()
	Gram, status = MVU(data, n_neighbors=3)

	from mvu_aux import save_gram, plot_MVU
	save_gram(Gram, f"resources/{dataset}_ori.npy")
	plot_MVU(Gram, dataset)