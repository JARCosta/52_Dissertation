import numpy as np

from scipy.sparse import csgraph, csr_matrix
from scipy.spatial.distance import cdist

from plot import plot

class ENG:
    def _neigh_matrix(self, X):
        def get_eta_d(Y, d):
            k = self.n_neighbors

            Y_neighs = self.k_neigh(X)[1]
            Y_neighs[np.where(Y_neighs == 0)] = np.inf
            
            eta_d = np.zeros((Y.shape[0]))
            for i in range(Y.shape[0]):
                Y_i = Y[np.argsort(Y_neighs[i])[:k]]
                U, sigma_i, Vh = np.linalg.svd(Y_i - Y[i] @ np.ones((Y.shape[1], 1)))

                eta_d[i] = np.sum(sigma_i[range(d)]) / np.sum(sigma_i[range(min(X.shape[1], k))])
            
            return np.sum(eta_d) / Y.shape[0]

        def connections(comp1, comp2, d=2, xi=0.95):
            Yp, Yq = X[comp1], X[comp2]
            s = min(Yp.shape[0], Yq.shape[0])
            
            dist_matrix = cdist(Yp, Yq)  # Compute distances

            connections = np.zeros((s, 2)).astype(int)
            for u in range(s):
                a, b = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
                connections[u] = [a, b]
                # print(dist_matrix[a][b])
                dist_matrix[a][b] = np.inf
            
            eta_d = get_eta_d(X, d)

            for l in range(d+1, (s)+1):
                a, b = connections[:l].T # 2xl, closest l connections between p and q
                # print(Yp[a], Yq[b])
                delta_l = Yp[a] - Yq[b]
                # print(delta_l)
                
                U, sigma_l, Vh = np.linalg.svd(delta_l)
                # print(U, sigma_l, Vh)
                
                eta_dl = np.sum(sigma_l[range(d)]) / sum(sigma_l[range(min(X.shape[1], l))])
                # print(sigma_l[range(d)], sigma_l[range(min(X.shape[1], l))])
                # print(eta_dl)

                l_max = l-1

                if eta_dl < xi * eta_d:
                    # print(l_max)
                    break 
            
            a, b = connections[:l_max].T
            return np.vstack((comp1[a], comp2[b])).T

        neigh_graph, neigh_matrix = self.k_neigh(X)[0:2]
        if self.model_args['plotation']:
            plot(X, neigh_matrix, title=f"initial {self.model_args['model']} data", block=False)
        cc = csgraph.connected_components(neigh_graph)

        while cc[0] > 1:
            # print(cc[0])
            components = []
            for i in range(cc[0]):
                idx = np.where(cc[1] == i)[0]
                components.append(idx)
                # print(X[idx].shape, neigh_matrix[idx][:, idx].shape)
                if self.model_args['plotation']:
                    plot(X[idx], neigh_matrix[idx][:, idx], title=f"component {i}", block=False)
            
            for i in range(cc[0]):
                comp1 = components[i]
                for j in range(cc[0]):
                    if j != i:
                        comp2 = components[j]
                        p1, p2 = connections(comp1, comp2).T
                        # print(np.vstack((p1, p2, np.sqrt(np.sum(np.square(X[p2]-X[p1]), axis=1)))).T)
        
                        for a, b in zip(p1, p2):
                            dist = np.linalg.norm(X[a] - X[b])
                            neigh_matrix[a, b] = dist
                            # neigh_matrix[b, a] = dist

                        # for a, b, d in np.vstack((p1, p2, np.sqrt(np.sum(np.square(X[p2]-X[p1]), axis=1)))).T:
                        #     neigh_matrix[int(a)][int(b)] = d
            neigh_graph = csr_matrix(neigh_matrix)
            cc = csgraph.connected_components(neigh_graph)

        return neigh_matrix
