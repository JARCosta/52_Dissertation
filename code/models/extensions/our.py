import numpy as np

from scipy.linalg import pinv
from scipy.sparse import csgraph, csr_matrix

from utils import stamp
from plot import plot

class Our:

    def __init__(self, bidirectional:bool):
        self.bidirectional = bidirectional

    def inter_connections(self, X_list:np.ndarray):

        C = np.zeros((len(X_list), len(X_list))) - 1
        for i in range(len(X_list)):
            for j in range(len(X_list)):
                if i != j:
                    r = X_list[j].mean(0)
                    closest_to_j_mean_coords = X_list[i][np.argmin(np.diag((X_list[i] - r) @ (X_list[i] - r).T))]
                    C[i][j] = np.where(np.all(X_list[i] == closest_to_j_mean_coords, axis=1))[0]
        return C.astype(int)


    def intra_connections(self, S_list:np.ndarray, C:np.ndarray):

        R:list[int] = [[] for _ in range(len(S_list))]
        for i in range(len(S_list)):
            A = np.zeros(len(S_list[i]))

            # Consider the already used inter-connection points
            for j in range(len(C[i])):
                if C[i][j] != -1:
                    with np.errstate(divide='ignore'):
                        idx = int(C[i][j])
                        A += np.log(np.diag((S_list[i] - S_list[i][idx]) @ (S_list[i] - S_list[i][idx]).T))

            # Get the rest reference points from intra-connections
            for j in range(len(C[i])-1, len(S_list[i][0])+1):
                r_idx = np.argmax(A)
                with np.errstate(divide='ignore'):
                    A += np.log(np.diag((S_list[i] - S_list[i][r_idx]) @ (S_list[i] - S_list[i][r_idx]).T))
                R[i].append(r_idx)
        return R

    def get_reference_points(self, X:np.ndarray, S:np.ndarray, C_global_index_list:list[list[int]]):
        """Get the reference dataset and its neighbourhood graph.
        
        Args:
            X (np.ndarray): The global input data points.
            S (np.ndarray): The global dataset of the flattened (individually) components.
            C_global_index_list (list[list[int]]): For each component, the list of global indexes of the corresponding points.
        """

        n_components = len(C_global_index_list)

        X_list = [X[C_global_index_list[i]] for i in range(n_components)]
        S_list = [S[C_global_index_list[i]] for i in range(n_components)]

        # the indexes inside the output of these functions are component wise indexes, not global indexes
        C = self.inter_connections(X_list)
        R = self.intra_connections(S_list, C)
        # print(C, R)
  

        reference_global_indexes_list = []
        X_reference_list = []
        C_reference_list = []
        for c in range(n_components):
            component_indexes = np.unique(np.concatenate([C[c], R[c]]))
            component_indexes = component_indexes[component_indexes != -1]
            component_global_indexes = C_global_index_list[c][component_indexes]
            
            reference_global_indexes_list.append(component_global_indexes.tolist())
            X_reference_list.append(X[reference_global_indexes_list[c]])
            C_reference_list.append([c,] * len(component_global_indexes))

        reference_global_indexes = np.concatenate(reference_global_indexes_list, axis=0)
        X_reference = np.concatenate(X_reference_list, axis=0)
        C_reference = np.concatenate(C_reference_list, axis=0)
        # print(X_reference, reference_global_indexes, C_reference)

        NM_reference = np.zeros((len(X_reference), len(X_reference)))
        for c_i in range(n_components):
            for c_j in range(n_components):
                if c_i != c_j:
                    # inter-component connections
                    a_global_index = C_global_index_list[c_i][C[c_i][c_j]] # _a_ point's index in the global dataset
                    b_global_index = C_global_index_list[c_j][C[c_j][c_i]]
                    # print(a_global_index, b_global_index)
                    a_reference_index = list(reference_global_indexes).index(a_global_index) # _a_ point's index in the reference dataset
                    b_reference_index = list(reference_global_indexes).index(b_global_index)
                    # print(a_reference_index, b_reference_index)
                    a_pos = X_reference[a_reference_index]
                    b_pos = X_reference[b_reference_index]

                    NM_reference[a_reference_index, b_reference_index] = (X_reference[a_reference_index] - X_reference[b_reference_index]) @ (X_reference[a_reference_index] - X_reference[b_reference_index]).T
                    if not self.bidirectional:
                        NM_reference[b_reference_index, a_reference_index] = NM_reference[a_reference_index, b_reference_index]

            # intra-component connections
            i_reference_global_indexes = reference_global_indexes_list[c_i]
            # print(i_reference_global_indexes)
            for j in range(len(i_reference_global_indexes)):
                for k in range(j+1, len(i_reference_global_indexes)):

                    a_global_index = i_reference_global_indexes[j] # _a_ point's index in the global dataset
                    b_global_index = i_reference_global_indexes[k]
                    # print(a_global_index, b_global_index)
                    a_component_index = list(C_global_index_list[c_i]).index(a_global_index) # _a_ point's index in the component dataset
                    b_component_index = list(C_global_index_list[c_i]).index(b_global_index)
                    # print(a_component_index, b_component_index)
                    a_pos = S_list[c_i][a_component_index]
                    b_pos = S_list[c_i][b_component_index]

                    a_reference_index = list(reference_global_indexes).index(a_global_index) # _a_ point's index in the reference dataset
                    b_reference_index = list(reference_global_indexes).index(b_global_index)

                    NM_reference[a_reference_index, b_reference_index] = (a_pos - b_pos) @ (a_pos - b_pos).T
                    if not self.bidirectional:
                        NM_reference[b_reference_index, a_reference_index] = NM_reference[a_reference_index, b_reference_index]
        # print("NM_reference", NM_reference)
        # print("Reference points:", reference_global_indexes)

        return reference_global_indexes, NM_reference, C_reference
    


    def fit_transform(self, X: np.ndarray):
        """Fits the model and computes embeddings."""
        
        temp, self.model_args['plotation'] = self.model_args['plotation'], False
        
        stamp.set()
        NM = super()._neigh_matrix(X)
        cc = csgraph.connected_components(NM)
        if cc[0] == 1:
            return NM
        stamp.print(f"*\t {self.model_args['dataname']}\t connected_components: {cc[0]}")

        
        C_global_index_list = [[] for _ in range(cc[0])] # [indexes of points corresponding to component i, for i in n_components]
        S_list = [[] for _ in range(cc[0])]
        for i in range(cc[0]):
            C_global_index_list[i] = np.where(cc[1] == i)[0]
            
            super().fit_transform(X[C_global_index_list[i]])
            
            S_list[i] = self.embedding_

        d = np.max([S_i.shape[1] for S_i in S_list])
        S = np.zeros((len(X), d))
        for i in range(cc[0]):
            S[C_global_index_list[i]] = S_list[i]

        reference_global_indexes, NM_reference, C_reference = self.get_reference_points(X, S, C_global_index_list)
        self.NM = NM_reference
        S_reference = super().fit(X[reference_global_indexes]).transform()

        for c_i in range(cc[0]):
            S_global_reference_i = S_reference[C_reference == c_i] # position of component's REFERENCE points in the GLOBAL linearization
            S_i = S[C_global_index_list[c_i]] # position of ALL component's points in the LOCAL linearization
            S_local_reference_i = S[reference_global_indexes[C_reference == c_i]] # position of component's REFERENCE points in the LOCAL linearization

            

            reference_mask = np.zeros((len(S_i),))
            print(reference_global_indexes[C_reference == c_i])
            for i in range(len(reference_global_indexes[C_reference == c_i])):
                reference_i_global_index = reference_global_indexes[C_reference == c_i][i]

                reference_i_local_index = np.where(C_global_index_list[c_i] == reference_i_global_index)[0][0]
                print(reference_i_global_index, reference_i_local_index)


                # reference_index = list(reference_global_indexes).index(reference_global_indexes[C_reference == c_i][i])
                reference_mask[reference_i_local_index] = 1
            
            plot(S_i, c=~reference_mask.astype(bool), block=False, title=f"Component {c_i} local linearization")

            transformation = pinv(S_local_reference_i) @ S_global_reference_i # transformation matrix from LOCAL to GLOBAL linearization
            S_i = S_i @ transformation # apply the transformation to the LOCAL linearization of the component
            S[C_global_index_list[c_i]] = S_i # update the global linearization with the transformed component

            plot(S_i, c=~reference_mask.astype(bool), block=False, title=f"Component {c_i} local linearization transformed")
        self.embedding_ = S
        
        self.model_args['plotation'] = temp
        if self.model_args['plotation']:
            reference_mask = np.zeros((len(X),))
            reference_mask[reference_global_indexes] = 1
            
            global_NM = np.zeros((len(X), len(X)))
            for i in range(len(reference_global_indexes)):
                i_global_index = reference_global_indexes[i]
                for j in range(len(reference_global_indexes)):
                    j_global_index = reference_global_indexes[j]
                    global_NM[i_global_index, j_global_index] = NM_reference[i, j]
            
            plot(X, global_NM, c=~reference_mask.astype(bool), title=f"{self.model_args['model']} input", block=False)
            plot(S, global_NM, c=~reference_mask.astype(bool), title=f"{self.model_args['model']} output", block=False)

        return self.embedding_



    