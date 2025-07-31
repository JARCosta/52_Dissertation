import numpy as np
import multiprocessing
import threading

from scipy.sparse import csgraph
import scipy.spatial as sp

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


from scipy.spatial import ConvexHull


def hull(points_A: np.ndarray, points_B: np.ndarray):
    """
    Calculate the overlap between two sets of points.
    
    Parameters
    ----------
        points_A : np.ndarray
            First set of points.
        points_B : np.ndarray
            Second set of points.

    Returns
    -------
        overlap : float
            Overlap between the two sets of points.
    """

    def in_hull(points, hull):
        # If the hull is None (e.g., due to not enough points), no points can be inside
        if not isinstance(hull, ConvexHull):
            return np.zeros(len(points), dtype=bool)
        # If hull has no facets (e.g., single point or line in higher dim), no points can be inside
        if hull.equations.shape[0] == 0:
            return np.zeros(len(points), dtype=bool)

        # Check if points satisfy all hyperplane inequalities
        # Add a small tolerance to the inequality check due to floating point precision
        # A common value for `eps` in Qhull is 1e-9 to 1e-15, so 1e-8 or 1e-9 is reasonable
        return np.all(np.dot(points, hull.equations[:, :-1].T) + hull.equations[:, -1] <= 1e-8, axis=1)
    
    # Determine the dimensionality from the input points
    # Assuming points_A and points_B have the same dimensionality
    dim = points_A.shape[1]

    # Ensure enough unique points for ConvexHull
    points_A_unique = np.unique(points_A, axis=0)
    points_B_unique = np.unique(points_B, axis=0)

    # Qhull requires at least d+1 points for a d-dimensional hull
    # Handle cases where components might be too small or degenerate
    if len(points_A_unique) < dim + 1 or len(points_B_unique) < dim + 1:
        utils.warning(f"Not enough unique points ({len(points_A_unique)} or {len(points_B_unique)}) for a {dim}-dimensional ConvexHull. Returning 0 overlap.")
        return 0.0 # Return 0 overlap if a hull cannot be formed

    try:
        # Compute Convex Hulls
        # Use QJ (joggle) and QbB (scale to unit cube) to improve robustness against precision errors
        hull_A = ConvexHull(points_A_unique, qhull_options="QJ QbB")
        hull_B = ConvexHull(points_B_unique, qhull_options="QJ QbB")

        # --- Calculate Overlap for Higher Dimensions (4D in your case) ---
        # We'll use the "percentage of points from one hull inside another" as the overlap metric.
        # This is a robust and commonly used proxy for overlap in higher dimensions.

        # Count points from A that are inside B's hull
        points_A_in_B_mask = in_hull(points_A_unique, hull_B)
        points_A_in_B_count = np.sum(points_A_in_B_mask)

        # Count points from B that are inside A's hull
        points_B_in_A_mask = in_hull(points_B_unique, hull_A)
        points_B_in_A_count = np.sum(points_B_in_A_mask)

        # Calculate overlap metrics (examples):

        # 1. Simple ratio of "shared" points to total unique points
        # total_unique_points_combined = len(points_A_unique) + len(points_B_unique)
        # overlap_ratio = (points_A_in_B_count + points_B_in_A_count) / total_unique_points_combined
        overlap_ratio = points_A_in_B_count / len(points_A_unique)


        # You can choose which metric to return. The Jaccard-like idea
        # of (intersection of points) / (union of points) is good.
        # For simplicity, let's use the average percentage of points from one hull in the other.
        # Or you could just return `overlap_ratio` as a single metric.

        # Print detailed overlap information
        # print(f"Number of points in A: {len(points_A_unique)}")
        # print(f"Number of points in B: {len(points_B_unique)}")
        # print(f"Points from A inside B's hull: {points_A_in_B_count} ({points_A_in_B_count/len(points_A_unique)*100 if len(points_A_unique)>0 else 0:.2f}%)")
        # print(f"Points from B inside A's hull: {points_B_in_A_count} ({points_B_in_A_count/len(points_B_unique)*100 if len(points_B_unique)>0 else 0:.2f}%)")
        # print(f"Combined points overlap ratio: {overlap_ratio:.4f}")

        # Decide what value your `overlap[i, j]` should store.
        # Let's return the combined points overlap ratio for the `overlap` matrix.
        return overlap_ratio

    except Exception as e:
        print(f"An error occurred during hull or overlap calculation: {e}")
        # import traceback
        # traceback.print_exc() # Uncomment to see the full traceback for debugging
        return 0.0 # Return 0.0 if an error occurs

class Based:
    def __init__(self, k1:int, k2:int):
        self.k1 = k1
        self.k2 = 2

    def NG(self, X:np.ndarray, k1:int, Ls:np.ndarray) -> list[np.ndarray] | None:
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
        cc, labels = csgraph.connected_components(NG, directed=False)
        if cc == 1:
            utils.warning("The data is already connected. Running default model.")
            self.model_args['artificial_connected'] = False
            return None, None
        
        Xl:list[np.ndarray] = [X[labels == c] for c in range(cc)]
        Ll:list[np.ndarray] = [Ls[labels == c] for c in range(cc)]
        return Xl, Ll

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

        # def DR(self, Xi:np.ndarray, Yl_results, i):
        #     Yi = super().fit_transform(Xi)
        #     if Yi is None:
        #         Yl_results[i] = None
        #         return

        #     Yi_dist = sp.distance.cdist(Yi, Yi)
        #     Xi_dist = sp.distance.cdist(Xi, Xi)
        #     dist_ratio = np.ones((len(Xi), len(Xi)))
        #     dist_ratio[Xi_dist != 0] = Xi_dist[Xi_dist != 0] / Yi_dist[Xi_dist != 0]
        #     if self.model_args['verbose']:
        #         print(f"X{i}/Y{i} ratio: {np.mean(dist_ratio)}")
        #     Yi = Yi * np.mean(dist_ratio)
        #     Yl_results[i] = Yi

        for i, Xi in enumerate(Xl):
            utils.important(f"linearising component {i}/{len(Xl)} ({len(Xi)})")
            if self.model_args['threaded']:
                # Yl_process = multiprocessing.Process(target=DR, args=[self, Xi, Yl_results, i])
                # utils.stamp.print(f"*\t {self.model_args['model']}\t launching process\t {Yl_process}")
                # Yl_processes.append(Yl_process)
                # Yl_process.start()
                pass
            else:
                self.eps = 1e-30
                Yi = super().fit_transform(Xi)
                if Yi is None:
                    return None

                if self.model_args['status'] != 'Solved':
                    utils.hard_warning(f"MVU didn't solve optimally({self.model_args['status']}). Recovering ratio.")
                    
                    Yi_NM = utils.neigh_matrix(Yi, self.k1)
                    min_Yi, max_Yi = np.min(Yi_NM[Yi_NM != 0]), np.max(Yi_NM)


                    Xi_NM = utils.neigh_matrix(Xi, self.k1)
                    min_Xi, max_Xi = np.min(Xi_NM[Xi_NM != 0]), np.max(Xi_NM)
                    # recovering_ratio = np.mean([min_Xi/min_Yi, max_Xi/max_Yi])
                    recovering_ratio = 10**(round(np.log10(max_Xi/max_Yi)))
                    
                    Yi = Yi * recovering_ratio

                    if self.model_args['verbose']:
                        print()
                        print(f"original\n\t min: {min_Xi:.4f}, max: {max_Xi:.4f}")
                        print(f"before recovery scalling\n\t min: {np.min(Yi_NM[Yi_NM != 0]):.4f}, max: {np.max(Yi_NM):.4f}")
                        print(f"recovery X{i}/Y{i} ratio: {recovering_ratio:.4f}")
                        Yi_NM = utils.neigh_matrix(Yi, self.k1)
                        print(f"after recovery scalling\n\t min: {np.min(Yi_NM[Yi_NM != 0]):.4f}, max: {np.max(Yi_NM):.4f}")
                        print()



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

    def chose_representative_points_iterative(self, Yl:list[np.ndarray], d:list[int], M:list[list[int]] = None):
        """
        For each component `Yi`, return the componentwise indexes of the representative points `Mi`.
        
        The next representative point is chosen as the point that is furthest from the current representative points (Sum distances).

        Parameters
        ----------
            Yl : list[np.ndarray]
                Componentwise local embeddings.
            d : list[int]
                Componentwise dimension.
            M : list[list[int]], optional
                Componentwise representative points.

        Returns
        -------
            M : list[list[int]]
                Componentwise representative points.
        """

        if M is None:
            M = [[] for _ in range(len(Yl))]

        def get_next_representative(Yi:np.ndarray, Mi:list[int]):
            if len(Mi) > 0:
                # dist_matrix = np.ones((Yi.shape[0], )) * np.inf
                dist_matrix = np.zeros((Yi.shape[0], ))
                
                for ma in Mi:
                    temp_dist = np.linalg.norm(Yi - Yi[ma], axis=1)
                    
                    # dist_matrix = np.minimum(dist_matrix, temp_dist)
                    dist_matrix = dist_matrix + temp_dist # picking duplicates
                    dist_matrix[Mi] = -np.inf
                
                return np.argmax(dist_matrix)
            else: # TODO: decide if we want to use the closest or the furthest from center
                return np.argmax(np.linalg.norm(Yi, axis=1))

        # M:list[list[int]] = [[] for _ in range(len(Yl))]
        for i, Yi in enumerate(Yl):
            for _ in range(len(M[i]), d[i] + 1):
                M[i].append(get_next_representative(Yi, M[i]))
        return M

    def chose_representative_points_PCA(self, Yl:list[np.ndarray], d:list[int], M:list[list[int]] = None):
        """
        For each component `Yi`, return the componentwise indexes of the representative points `Mi`.
        
        Apply PCA to each component `Yi` and chose the two points that are furthest from the center, for each dimension `d[i]`.

        Parameters
        ----------
            Yl : list[np.ndarray]
                Componentwise local embeddings.
            d : list[int]
                Componentwise dimension.
            M : list[list[int]], optional
                Componentwise representative points.

        Returns
        -------
            M : list[list[int]]
                Componentwise representative points.
        """
        
        if M is None:
            M = [[] for _ in range(len(Yl))]

        for i, Yi in enumerate(Yl):
            kernel = Yi @ Yi.T
            eigenvalues, eigenvectors = np.linalg.eigh(kernel)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:, idx]
            eigenvalues = eigenvalues[:d[i]]
            eigenvectors = eigenvectors[:, :d[i]]
            restricted_embedding = eigenvectors @ np.diag(eigenvalues)
            

            max_idx = np.argmax(restricted_embedding, axis=0)
            M[i].extend([int(a) for a in max_idx])
            min_idx = np.argmin(restricted_embedding, axis=0)
            M[i].extend([int(a) for a in min_idx])
        return M



    def chose_intercomponent_connections_default(self, Xl:list[np.ndarray], M:list[list[int]]):
        """
        Iteratively connect the largest component to the next closest component.

        Parameters
        ----------
            Xl : list[np.ndarray]
                Componentwise original points.
            M : list[list[int]]
                Componentwise representative points.

        Returns
        -------
            L : list[list[list[int]]]
                Inter-component connections.
        """

        def global_to_local(global_idx:int):
            i, loc_idx = 0, global_idx
            while loc_idx >= len(Xl[i]):
                loc_idx -= len(Xl[i])
                i += 1
            return i, loc_idx

        L:list[list[list[int]]] = [] # L.append([[0, 345], [2 , 741], 34.23])

        Xs = np.vstack(Xl)
        NG = utils.neigh_graph(Xs, self.k1)
        cc, labels = csgraph.connected_components(NG, directed=False)

        while cc > 1:
            largest_component = np.argmax(np.bincount(labels))
            largest_component_idx = np.where(labels == largest_component)[0]
            other_idx = np.where(labels != largest_component)[0]

            distances = sp.distance.cdist(Xs[largest_component_idx], Xs[other_idx])
            shortest_distance = np.min(distances)
            ab = np.where(distances == shortest_distance)

            a_idx, b_idx = largest_component_idx[ab[0]], other_idx[ab[1]]
            a_idx, b_idx = int(a_idx), int(b_idx)
            NG[a_idx, b_idx] = 1

            i, p = global_to_local(a_idx)
            j, q = global_to_local(b_idx)

            if self.model_args['verbose']:
                print(f"{p}/{len(Xl[i])} ({i}) -> {q}/{len(Xl[j])} ({j})")

            if p not in M[i]:
                M[i].append(p)
            if q not in M[j]:
                M[j].append(q)
            p = M[i].index(p)
            q = M[j].index(q)

            L.append([[i, p], [j, q], shortest_distance])

            cc, labels = csgraph.connected_components(NG, directed=False)
        return L

    def chose_intercomponent_connections_k2(self, Xl:list[np.ndarray], M:list[list[int]], k2:int):
        """
        For each component `Xi`, chose the `k2` connections to the other components (representative points representation).

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
                    p, q = int(p), int(q)
                    L.append([[i, p], [j, q], dist_matrix[p, q]])

                    connections = [L_ij for L_ij in L if L_ij[0][0] == i]
                    if len(connections) > k2:
                        connection_distances = [L_ij[2] for L_ij in connections]
                        argmax_idx = np.argmax(connection_distances)
                        L.remove(connections[argmax_idx])
        return L

    def final_MVU(self, Xl:list[np.ndarray], Yl:list[np.ndarray], M:list[list[int]], L:list[list[list[int]]], d:int):
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
            NM[ptr:ptr+len(YMi), ptr:ptr+len(YMi)] = utils.neigh_matrix(YMi, len(YMi) - 1)
        
        # inter-component connections (distances in original space)
        for [i, p], [j, q], dist in L:
            a = np.sum([len(XM[k]) for k in range(i)]) + p
            b = np.sum([len(XM[k]) for k in range(j)]) + q
            a, b = int(a), int(b)
            NM[a, b] = np.linalg.norm(XMs[b] - XMs[a])
            if dist != NM[a, b]: # cdist != norm
                cdist = sp.distance.cdist(XMs[[a, ]], XMs[[b, ]])
                print(f"dist: {dist}, cdist: {cdist}, norm: {NM[a, b]}")
        self.NM = NM
        

        if self.model_args['verbose']:
            # for i, XMi in enumerate(XM):
            #     print(f"X{i} in global space:")
            #     print(XMi)
            
            # print("Final MVU NM:")
            # for NMi in NM:
            #     print([int(a) for a in NMi])
            
            print(f"NM.shape: {NM.shape}")
            print(f"NM.max(): {np.max(NM)}")
        
        utils.stamp.print(f"*\t {self.model_args['model']}\t global MVU")
        # self.model_args['verbose'] = True
        self.n_components = None # sup(d(components)) - (#components - 1)
        self.eps = 1e-15
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

        # if self.model_args['verbose']:
        #     for i, YMi in enumerate(YM):
        #         print(f"Y{i} in global space:")
        #         print(YMi)

        return YM


    def equalize_dimensions(self, Yl:list[np.ndarray]):
        """
        Equalize the dimensions of the local embeddings.
        """

        max_dim = max([Yi.shape[1] for Yi in Yl])
        return [np.hstack((Yi, np.zeros((Yi.shape[0], max_dim - Yi.shape[1])))) for Yi in Yl]

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
            # M[i]'s position in local embeddings @ Ti      = M[i]'s position in the global embedding space
            # YMi                                 @ Tl[i]   = YM[i]
            
            
            # Ax + b = Y
            # [A, b] @ [x, 1] = Y
            # [A, b] = [X, 1]^-1 @ Y
            # T      = [X, 1]^-1 @ Y
            
            # YMi = Yi[M[i]] # X : local linearised M points
            # YM[i]          # Y : global lineasired points

            # # Ti = np.linalg.pinv(np.hstack((YMi, np.ones((len(YMi), 1)) ))) @ YM[i]
            # Ti, _, _, _ = np.linalg.lstsq(np.hstack((YMi, np.ones((len(YMi), 1)))), YM[i], rcond=None)
            # Tl.append(Ti)
            
            
            P = Yi[M[i]]
            Q = YM[i]

            centroid_P = P.mean(axis=0)
            centroid_Q = Q.mean(axis=0)

            P_centered = P - centroid_P
            Q_centered = Q - centroid_Q

            H = P_centered.T @ Q_centered

            U, S, Vt = np.linalg.svd(H)

            R = Vt.T @ U.T
            
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T

            s = np.sum(S) / np.sum(P_centered**2)

            t = centroid_Q - s * R @ centroid_P

            Tl.append([s, R, t])
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
            # Yl[i] = np.hstack((Yi, np.ones((Yi.shape[0], 1)))) @ Tl[i]

            s, R, t = Tl[i]
            original_points = Yi
            transformed_points = (s * (R @ original_points.T)).T + t
            Yl[i] = transformed_points
        return Yl





















    ########################################################
    # EXTEND ###############################################
    ########################################################

    def fit_transform(self, X:np.ndarray):
        Xs = X
        k1, k2 = self.k1, self.k2

        if 'labels' not in self.model_args:
            labels = np.zeros(Xs.shape[0])
        else:
            labels = self.model_args['labels']
        Xl, Ll = self.NG(Xs, k1, labels)
        if Xl is None: # might be useless, it runs good without it, computes the global embeddings unnecessarilly
            return super().fit_transform(Xs)
        labels = np.vstack(Ll)
        Xs = np.vstack(Xl)

        # if self.model_args['plotation']:
        #     color = [i for i in range(len(Xl)) for _ in range(len(Xl[i]))]
        #     plot.plot(Xs, c=color, block=False, title=f"{self.model_args['dataname']}({self.model_args['#neighs']}) MVU local stage")


        self.ratio = None # toggle compute independent ratio
        Yl = self.MVU_local(Xl)
        if Yl is None:
            return None
        
        if self.model_args['verbose']:
            print("Before transformation:")
            for i, Yi in enumerate(Yl):
                Xi, Li, Yi = Xl[i], Ll[i], Yl[i]
                print(f"Component {i}({len(Xi)}/{len(Xs)}):")
                utils.compute_measures(Xi, Yi, Li, k1)

        d:list[int] = [Yi.shape[1] for Yi in Yl]
        M = [[] for _ in range(len(Yl))]
        M = self.chose_representative_points_PCA(Yl, d, M) # before any L
        L = self.chose_intercomponent_connections_default(Xl, M) # after PCA, before iterative
        # M = self.chose_representative_points_iterative(Yl, d, M) # after default, before k2
        # L = self.chose_intercomponent_connections_k2(Xl, M, k2) # after any M
        
        # if self.model_args['verbose']:
        #     print(f"M: {M}")
        # if self.model_args['verbose']:
        #     print(f"inter-component connections: {L}")


        Yl = self.equalize_dimensions(Yl)
        YM = self.final_MVU(Xl, Yl, M, L, max(d))
        if YM is None:
            return None

        d_diff = Yl[0].shape[1] - YM[0].shape[1]
        if d_diff > 0:
            YM = [np.hstack((YMi, np.zeros((YMi.shape[0], d_diff)))) for YMi in YM]
        elif d_diff < 0:
            Yl = [np.hstack((YMi, np.zeros((YMi.shape[0], -d_diff)))) for YMi in Yl]
        
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
            plot.plot_two(Xs, YMs, NM, NM2, scale=True, scale2=False, c1=color, c2=color2, block=False, title=f"{self.model_args['dataname']}({self.model_args['#neighs']}) global stage, X and Y")
        
        Tl = self.final_transformation(Yl, YM, M)
        Yl_old = Yl.copy()
        Yl = self.final_final_yeet(Yl, Tl)

        if self.model_args['verbose']:
            print("After transformation:")
            for i, Yi in enumerate(Yl):
                Xi, Li, Yi = Xl[i], Ll[i], Yl[i]
                print(f"Component {i}({len(Xi)}/{len(Xs)}):")
                utils.compute_measures(Xi, Yi, Li, k1)
            

            def compute_nn(Y:np.ndarray, L:np.ndarray, dim:int) -> np.ndarray:
                """
                Compute the nearest neighbor matrix for a given component.
                
                Parameters
                ----------
                    Y : np.ndarray
                        Embeddings.
                    L : np.ndarray
                        True labels.
                    dim : int
                        Dimension of the labels.
                
                Returns
                -------
                    i_individual_nn : np.ndarray
                        Nearest neighbor matrix.
                """

                temp_NG = utils.neigh_graph(Y, 1)
                nearest_neigh_idx = [temp_NG[p].indices[0] for p in range(Y.shape[0])]
                # print(Y.shape, L.shape, nearest_neigh_idx.shape)
                L_predicted = L[nearest_neigh_idx]
                
                individual_nn = np.zeros((dim, dim))
                for p in range(Y.shape[0]):
                    actual_label = int(L[p])
                    predicted_label = int(L_predicted[p])
                    individual_nn[actual_label, predicted_label] += 1
                return individual_nn
                



            dim = int(np.max(labels)) + 1
            for i, Yi in enumerate(Yl):
                Li = Ll[i]
                i_individual_nn = compute_nn(Yi, Li, dim)

                for j, Yj in enumerate(Yl):
                    if i < j:
                        Lj = Ll[j]
                        j_individual_nn = compute_nn(Yj, Lj, dim)
                        

                        print(f"component collisions {i} and {j}:")
                        individual_nn = i_individual_nn + j_individual_nn
                        
                        
                        
                        Ym, Lm = np.vstack((Yi, Yj)), np.vstack((Li, Lj))
                        merged_nn = compute_nn(Ym, Lm, dim)

                        diff = merged_nn - individual_nn
                        print(f"diff: \n{diff}")

                        color = np.zeros(Ym.shape[0])
                        color[Yi.shape[0]:] = 1
                        plot.plot(Ym, c=color, block=False, title=f"Component {i} and {j} merged")
            print("Global collisions:")
            
            individual_nn = np.zeros((dim, dim))
            for Yi, Li in zip(Yl, Ll):
                individual_nn += compute_nn(Yi, Li, dim)

            Ym, Lm = np.vstack(Yl), np.vstack(Ll)
            merged_nn = compute_nn(Ym, Lm, dim)

            print(f"merged_nn: \n{merged_nn}")
            print(f"individual_nn: \n{individual_nn}")
            print(f"diff: \n{merged_nn - individual_nn}")

            one_nn = individual_nn.copy()
            for i in range(dim):
                one_nn[i, i] = 0
            one_nn = np.sum(one_nn) / np.sum(individual_nn)
            print(f"individual_1_NN: \n{one_nn}")

            one_nn = merged_nn.copy()
            for i in range(dim):
                one_nn[i, i] = 0
            one_nn = np.sum(one_nn) / np.sum(merged_nn)
            print(f"merged_1_NN: \n{one_nn}")
            



            # print("Component overlap:")
            # overlap = np.zeros((len(Yl), len(Yl)))
            # for i, Yi in enumerate(Yl):
            #     for j, Yj in enumerate(Yl):
            #         if i != j:
            #             overlap[i, j] = hull(Yi, Yj)
            # print(overlap)
        
        Ys, Ls = np.vstack(Yl), np.vstack(Ll)

        if self.model_args['plotation']:
            for i in range(len(Yl)):
                # color = [0 if a in M[i] else 1 for a in range(len(Yl[i]))]
                Li = Ll[i]
                
                NM = np.zeros((len(Yl[i]), len(Yl[i])))
                for a in M[i]:
                    for b in M[i]:
                        if b > a:
                            NM[a, b] = 1
                plot.plot_two(Yl_old[i], Yl[i], NM, NM, c1=Li, c2=Li, c1_scale=[np.min(Ls), np.max(Ls)], c2_scale=[np.min(Ls), np.max(Ls)], block=False, title=f"Component {i}({len(Yl[i])}), before and after Translation and Rotation")

            NM = get_NMg(Yl, M, L)
            color1 = [0 if a in M[i] else 1 for i in range(len(Yl)) for a in range(len(Yl[i]))]
            color2 = [i for i in range(len(Yl)) for _ in range(len(Yl[i]))]
            plot.plot_two(Ys, Ys, NM, NM, scale=True, scale2=False, c1=color1, c2=color2, block=False, title=f"{self.model_args['dataname']}({self.model_args['#neighs']}) final stage")

        self.embedding_ = Ys
        self.model_args['labels'] = Ls
        return Ys