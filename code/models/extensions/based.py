import numpy as np
import multiprocessing
import threading

from scipy.sparse import csgraph
import scipy.spatial as sp


from utils import warning
import utils
import plot

np.set_printoptions(linewidth=200)


def get_NMg(Dl:list[np.ndarray], M:list[list[int]], L:list[list[list[int]]]):
    """
    Parameters
    ----------
        Dl : list[np.ndarray]
            Componentwise data points. (stackable/same dimensions)
        M : list[list[int]]
            Componentwise representative points.
        L : list[list[list[int]]]
            Inter-component connections.

    Returns
    -------
        NM : np.ndarray
            Global connection matrix.
    """

    Ds = np.vstack(Dl)
    NM = np.zeros((len(Ds), len(Ds)))
    for i in range(len(Dl)):
        ptr = np.sum([len(Dl[k]) for k in range(i)])
        ptr = int(ptr)
        for a in M[i]:
            for b in M[i]:
                if b > a:
                    NM[ptr+a, ptr+b] = 1

    for [i, p], [j, q], dist in L:
        a_idx = np.sum([len(Dl[k]) for k in range(i)]) + M[i][p]
        b_idx = np.sum([len(Dl[k]) for k in range(j)]) + M[j][q]
        a_idx, b_idx = int(a_idx), int(b_idx)
        NM[a_idx, b_idx] = 1

    return NM



class Based:
    def __init__(self, k1:int, k2:int):
        self.k1 = k1
        self.k2 = 2

    def NG(self, X:np.ndarray, k1:int):
        """
        Parameters
        ----------
            X : np.ndarray
                Original points.
            k1 : int
                Size of the k-neighbourhood.

        Returns
        -------
            Xl : list[np.ndarray]
                Componentwise original points.
        """

        NG = utils.neigh_graph(X, k1)
        cc = csgraph.connected_components(NG)
        if cc[0] == 1:
            warning("The data is already connected. Running default model.")
            self.model_args['artificial_connected'] = False
            return None
        
        Xl:list[np.ndarray] = [X[cc[1] == c] for c in range(cc[0])]
        return Xl

    def MVU_local(self, Xl:list[np.ndarray]):
        """
        Parameters
        ----------
            Xl : list[np.ndarray]
                Componentwise original points.

        Returns
        -------
            Yl : list[np.ndarray]
                Componentwise local embeddings.
        """

        Yl:list[np.ndarray] = []
        
        Yl_processes:list[multiprocessing.Process] = []
        Yl_results = multiprocessing.Manager().dict()

        def DR(Xi:np.ndarray, Yl_results, i):
            Yi = super().fit_transform(Xi)

            Yi_dist = sp.distance.cdist(Yi, Yi)
            Xi_dist = sp.distance.cdist(Xi, Xi)
            dist_ratio = np.ones((len(Xi), len(Xi)))
            dist_ratio[Xi_dist != 0] = Xi_dist[Xi_dist != 0] / Yi_dist[Xi_dist != 0]
            if self.model_args['verbose']:
                print(f"Y{i} ratio: {np.mean(dist_ratio)}")
            Yi = Yi * np.mean(dist_ratio)
            Yl_results[i] = Yi
            return Yi

        for i, Xi in enumerate(Xl):
            utils.stamp.print(f"*\t {self.model_args['model']}\t linearising component {i}({len(Xi)})")
            if self.model_args['threaded']:
                Yl_process = multiprocessing.Process(target=DR, args=[Xi, Yl_results, i])
                utils.stamp.print(f"*\t {self.model_args['model']}\t launching process\t {Yl_process}")
                Yl_processes.append(Yl_process)
                Yl_process.start()
            else:
                Yi = super().fit_transform(Xi)
                if Yi is None:
                    return None
                
                Yi_dist = sp.distance.cdist(Yi, Yi)
                Xi_dist = sp.distance.cdist(Xi, Xi)
                dist_ratio = np.ones((len(Xi), len(Xi)))
                dist_ratio[Xi_dist != 0] = Xi_dist[Xi_dist != 0] / Yi_dist[Xi_dist != 0]
                if self.model_args['verbose']:
                    print(f"Y{i} ratio: {np.mean(dist_ratio)}")
                Yi = Yi * np.mean(dist_ratio)
                Yl_results[i] = Yi

        if self.model_args['threaded']:
            [process.join() for process in Yl_processes]
        
        for i, Yi in Yl_results.items():
            Yl.append(Yi)

        # if self.model_args['verbose']:
        #     for i, Xi in enumerate(Xl):
        #         Xi_dist = sp.distance.cdist(Xi, Xi)
        #         for a in range(len(Xi)):
        #             Xi_dist[a][a] = np.median(Xi_dist)
        #         print(f"idx_min=({np.argmin(Xi_dist)//len(Xi)},{np.argmin(Xi_dist)%len(Xi)}), value={np.min(Xi_dist)}")
        #         print(f"idx_max=({np.argmax(Xi_dist)//len(Xi)},{np.argmax(Xi_dist)%len(Xi)}), value={np.max(Xi_dist)}")
                
        #         Yi = Yl[i]
        #         Yi_dist = sp.distance.cdist(Yi, Yi)
        #         for a in range(len(Yi)):
        #             Yi_dist[a][a] = np.median(Yi_dist)
        #         print(f"idx_min=({np.argmin(Yi_dist)//len(Yi)},{np.argmin(Yi_dist)%len(Yi)}), value={np.min(Yi_dist)}")
        #         print(f"idx_max=({np.argmax(Yi_dist)//len(Yi)},{np.argmax(Yi_dist)%len(Yi)}), value={np.max(Yi_dist)}")
        return Yl

    def chose_representative_points(self, Yl:list[np.ndarray], d:list[int]):
        """
        For each component `Yi`, return the componentwise indexes of the representative points `Mi`.
        
        The representative points are chosen as the points that are furthest from the center of the component.

        Parameters
        ----------
            Yl : list[np.ndarray]
                Componentwise local embeddings.
            d : list[int]
                Componentwise dimension.

        Returns
        -------
            M : list[list[int]]
                Componentwise representative points.
        """

        def get_next_representative(Yi:np.ndarray, Mi:list[int]):
            if len(Mi) > 0:
                dist_matrix = np.ones((Yi.shape[0], )) * np.inf
                # dist_matrix = np.zeros((Yi.shape[0], ))
                
                for ma in Mi:
                    temp_dist = np.linalg.norm(Yi - Yi[ma], axis=1)
                    
                    dist_matrix = np.minimum(dist_matrix, temp_dist)
                    # dist_matrix = dist_matrix + temp_dist # picking duplicates
                    # dist_matrix[Mi] = -np.inf
                
                return np.argmax(dist_matrix)
            else: # TODO: decide if we want to use the closest or the furthest from center
                return np.argmax(np.linalg.norm(Yi, axis=1))

        M:list[list[int]] = [[] for _ in range(len(Yl))]
        for i, Yi in enumerate(Yl):
            M[i] = []
            for _ in range(d[i]+1):
                M[i].append(get_next_representative(Yi, M[i]))
        return M



    def chose_intercomponent_connections(self, Xl:list[np.ndarray], M:list[list[int]], k2:int):
        """
        Parameters
        ----------
            Xl : list[np.ndarray]
                Componentwise original points.
            M : list[list[int]]
                Componentwise representative points.
            k2 : int
                Size of the inter-component k-neighbourhood.

        Returns
        -------
            L : list[list[list[int]]]
                Inter-component connections.
        """

        L:list[list[list[int]]] = [] # L.append([[0, 345], [2 , 741], 34.23])
        for i, Xi in enumerate(Xl):
            Mi = M[i]
            XMi = Xi[Mi]
            for j, Xj in enumerate(Xl):
                Mj = M[j]
                XMj = Xj[Mj]
                if j != i:
                    dist_matrix = sp.distance.cdist(XMi, XMj)
                    
                    p, q = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
                    L.append([[i, p], [j, q], dist_matrix[p, q]])

                    connections = [L_ij for L_ij in L if L_ij[0][0] == i]
                    if len(connections) > k2:
                        connection_distances = [L_ij[2] for L_ij in connections]
                        argmax_idx = np.argmax(connection_distances)
                        L.remove(connections[argmax_idx])
        return L

    def final_MVU(self, Xl:list[np.ndarray], Yl:list[np.ndarray], M:list[list[int]], L:list[list[list[int]]]):
        """
        Compute the global embeddings of the representative points.
        
        Parameters
        ----------
            Xl : list[np.ndarray]
                Componentwise original points.
            Yl : list[np.ndarray]
                Componentwise local embeddings.
            M : list[list[int]]
                Componentwise representative points.
            L : list[list[list[int]]]
                Inter-component connections.

        Returns
        -------
            YM : list[np.ndarray]
                Componentwise global embeddings of the representative points.
        """
        XM = [Xi[M[i]] for i, Xi in enumerate(Xl)]
        XMs = np.vstack(XM)

        YM = [Yi[M[i]] for i, Yi in enumerate(Yl)]

        NM = np.zeros((len(XMs), len(XMs)))
        # intra-component connections (distances from local embeddings)
        for i, YMi in enumerate(YM):
            ptr = np.sum([len(YM[j]) for j in range(i)])
            ptr = int(ptr)
            NM[ptr:ptr+len(YMi), ptr:ptr+len(YMi)] = sp.distance.cdist(YMi, YMi)
        
        # inter-component connections (distances in original space)
        for [i, p], [j, q], dist in L:
            a = np.sum([len(XM[k]) for k in range(i)]) + p
            b = np.sum([len(XM[k]) for k in range(j)]) + q
            a, b = int(a), int(b)
            NM[a, b] = sp.distance.cdist(XMs[[a, ]], XMs[[b, ]])
        self.NM = NM
        

        if self.model_args['verbose']:
            for i, XMi in enumerate(XM):
                print(f"X{i} in global space:")
                print(XMi)
            print("Final MVU NM:")
            for NMi in NM:
                print(list(NMi))
            

        if self.model_args['verbose']:
            print(f"NM.shape: {NM.shape}")
            print(f"NM.max(): {np.max(NM)}")
        
        self.model_args['verbose'] = True
        YMs = super().fit(XMs).transform()
        if YMs is None:
            return None
        assert XMs.shape[0] == YMs.shape[0], f"XMs.shape: {XMs.shape}, YMs.shape: {YMs.shape}"

        YM = []
        for i in range(len(Yl)):
            ptr = np.sum([len(M[k]) for k in range(i)])
            ptr = int(ptr)
            YMi = YMs[ptr:ptr+len(M[i])]
            
            YM.append(YMi)
        
        assert all(len(M[i]) == len(YM[i]) for i in range(len(M))), f"M: {M}, YM: {YM}"

        if self.model_args['verbose']:
            for i, YMi in enumerate(YM):
                print(f"Y{i} in global space:")
                print(YMi)

        if self.model_args['plotation']:
            plot.plot_two(XMs, YMs, NM, NM, scale=True, scale2=True, block=False, title=f"{self.model_args['dataname']}({self.model_args['#neighs']}) global stage, X and Y")

        return YM


    def final_transformation(self, Yl:list[np.ndarray], YM:list[np.ndarray], M:list[list[int]]):
        """
        Parameters
        ----------
            Yl : list[np.ndarray]
                Componentwise local embeddings.
            YM : list[np.ndarray]
                Componentwise global embeddings of the representative points.
            M : list[list[int]]
                Componentwise representative points.
        Returns
        -------
            Tl : list[np.ndarray]
                Componentwise transformation positioning the local embeddings in the global embedding space.
        """
        
        Tl = []
        for i, Yi in enumerate(Yl):
            YMi = Yi[M[i]]
            # M[i]'s position in local embeddings @ Ti      = M[i]'s position in the global embedding space
            # YMi                                 @ Tl[i]   = YM[i]
            
            # Ax + b = Y
            # [A, b] @ [x, 1] = Y
            # T = [X, 1]^-1 @ Y            
            Ti = np.linalg.pinv(np.hstack((YMi, np.ones((len(YMi), 1))))) @ YM[i]
            Tl.append(Ti)
        return Tl
    

    def final_final_yeet(self, Yl:list[np.ndarray], Tl:list[np.ndarray]):
        """
        Parameters
        ----------
            Yl : list[np.ndarray]
                Componentwise local embeddings.
            Tl : list[np.ndarray]
                Componentwise transformation positioning the local embeddings in the global embedding space.
        
        Returns
        -------
            Yl: list[np.ndarray]
                Componentwise embeddings.
        """

        for i, Yi in enumerate(Yl):
            Yl[i] = np.hstack((Yi, np.ones((Yi.shape[0], 1)))) @ Tl[i]
        return Yl





















    ########################################################
    # EXTEND ###############################################
    ########################################################

    def fit_transform(self, Xs:np.ndarray):

        k1, k2 = self.k1, self.k2
        
        Xl:list[np.ndarray] = self.NG(Xs, k1)
        if Xl is None: # might be useless, it runs good without it, computes the global embeddings unnecessarilly
            return super().fit_transform(Xs)

        temp = self.model_args['#components']
        self.model_args['#components'] = None
        Yl = self.MVU_local(Xl)
        if Yl is None:
            return None
        self.model_args['#components'] = temp

        d:list[int] = [Yi.shape[1] for Yi in Yl]
        M:list[np.ndarray] = self.chose_representative_points(Yl, d)

        if self.model_args['verbose']:
            for i, Yi in enumerate(Yl):
                print(f"Y{i} in local space:")
                print(Yi[M[i]])
        if self.model_args['plotation']:
            YM = [Yi[M[i]] for i, Yi in enumerate(Yl)]
            for i in range(len(Yl)):
                
                color = [0 if a in M[i] else 1 for a in range(len(Yl[i]))]
                
                NM = np.zeros((len(Yl[i]), len(Yl[i])))
                for a in M[i]:
                    for b in M[i]:
                        if b > a:
                            NM[a, b] = 1
                NM2 = np.ones((len(YM[i]), len(YM[i])))
                plot.plot_two(Yl[i], YM[i], NM, NM2, scale=False, c1=color, block=False, title=f"Component {i}({len(Yl[i])}), local and global")
        
        
        L = self.chose_intercomponent_connections(Xl, M, k2)
        if self.model_args['verbose']:
            utils.stamp.print(f"*\t {self.model_args['model']}\t inter-component connections: {L}")

        YM = self.final_MVU(Xl, Yl, M, L)
        if YM is None:
            return None
        
        if self.model_args['plotation']:
            YMs = np.vstack(YM)
            color2 = [i for i in range(len(YM)) for a in range(len(YM[i]))]

            NM = get_NMg(Xl, M, L)
            
            NM2 = np.zeros((len(YMs), len(YMs)))
            for i in range(len(YM)):
                ptr = np.sum([len(YM[k]) for k in range(i)])
                ptr = int(ptr)
                NM2[ptr:ptr+len(YM[i]), ptr:ptr+len(YM[i])] = 1
            
            for [i, p], [j, q], _ in L:
                a = np.sum([len(YM[k]) for k in range(i)]) + p
                b = np.sum([len(YM[k]) for k in range(j)]) + q
                a, b = int(a), int(b)
                NM2[a, b] = 1

            color = [0 if a in M[i] else 1 for i, Xi in enumerate(Xl) for a in range(len(Xi))]
            plot.plot_two(Xs, YMs, NM, NM2, scale=True, scale2=True, c1=color, c2=color2, block=False, title=f"{self.model_args['dataname']}({self.model_args['#neighs']}) global stage, X and Y")

            # for i, YMi in enumerate(YM):
            #     dist = sp.distance.cdist(YMi, YMi)
            #     print(f"Intra_distance, Y{i}:\n{dist}")
            #     plot.plot(YMi, np.ones((len(YMi), len(YMi))), scale=False, block=False, title=f"Component {i} in global space")
        
        Tl = self.final_transformation(Yl, YM, M)
        Yl_old = Yl.copy()
        Yl = self.final_final_yeet(Yl, Tl)
        Ys = np.vstack(Yl)

        if self.model_args['plotation']:
            for i in range(len(Yl)):
                plot.plot_two(Yl_old[i], Yl[i], scale=False, scale2=False, block=False, title=f"Component {i}({len(Yl[i])}), before and after Translation and Rotation")

            NM = get_NMg(Yl, M, L)
            color = [0 if a in M[i] else 1 for i in range(len(Yl)) for a in range(len(Yl[i]))]
            plot.plot_two(Ys, Ys, NM, NM, scale=True, scale2=False, c1=color, c2=color, block=False, title=f"{self.model_args['dataname']}({self.model_args['#neighs']}) final stage")

        self.embedding_ = Ys
        return Ys