import numpy as np
import sklearn.manifold

from utils import k_neigh, load_cache, save_cache, stamp
import utils

def TC(X, Y, n_neighbors) -> tuple[float, float]:
    import scipy.spatial
    
    n_samples = X.shape[0]
    
    NM_X = k_neigh(X, n_neighbors, reduction=None)[1] # need symetrix
    NM_Y = k_neigh(Y, n_neighbors, reduction=None)[1] # need symetrix

    NM_X = np.where(NM_X != 0, 1, 0)
    NM_Y = np.where(NM_Y != 0, 1, 0)

    NM_T = NM_Y - NM_X

    NM_T[np.where(NM_T == -1)] = 0
    D_X = utils.intra_dist_matrix(X)
    R_X = np.argsort(np.argsort(D_X, axis=1), axis=1)
    T = NM_T * R_X
    T[T != 0] -= n_neighbors
    T = 1 - np.sum(T) * (2 / (n_samples * n_neighbors * (2*n_samples - 3*n_neighbors - 1)))



    NM_C = NM_X - NM_Y
    NM_C[np.where(NM_C == -1)] = 0
    D_Y = utils.intra_dist_matrix(Y)
    R_Y = np.argsort(np.argsort(D_Y, axis=1), axis=1)
    C = NM_C * R_Y
    C[C != 0] -= n_neighbors
    C = 1 - np.sum(C) * (2 / (n_samples * n_neighbors * (2*n_samples - 3*n_neighbors - 1)))



    # print(T, sklearn.manifold.trustworthiness(X, Y, n_neighbors=n_neighbors))
    # breakpoint()



    return round(float(T), 3), round(float(C), 3)

def one_NN(Y, labels) -> float:
    if labels is None:
        utils.warning("labels is None")
        return None
    
    _, _, IM = k_neigh(Y, 1, reduction=False) # need symetrix

    Y_labels = np.zeros(labels.shape)
    for i in range(Y.shape[0]):
        Y_labels[i] = labels[IM[i]]
    
    one_NN = np.count_nonzero(Y_labels - labels) / labels.shape[0]
    return round(float(one_NN), 3)


if __name__ == "__main__":

    for dataname in ['swiss', 'brokenswiss', 'changing_swiss', 'helix', 'twinpeaks', '3d_clusters', 'intersect', 'difficult']:
        model_args = {'dataname': dataname, '#points': 1000}
        X:np.ndarray = load_cache(model_args, "X")
        labels:np.ndarray = load_cache(model_args, "l")
        t:np.ndarray = load_cache(model_args, "t")

        from plot import plot
        plot(X, block=False, c=labels, title=str(model_args))

        if X is None:
            input(f"TODO: X[{dataname}] = None")
        
        for model in ["isomap", "mvu", "mvu.ineq"]:
            for n_neighbors in range(3, 5):

                model_args = {
                    k: v for k, v in {
                        "dataname": dataname,
                        "#points": 1000,
                        '#neighs': n_neighbors,
                        "model": model,
                        "#components": 50 if model == "isomap" else None,
                        "eps": 1e-3 if model in ["mvu", "mvu.ineq"] else None,
                    }.items() if v is not None
                }


                Y:np.ndarray = load_cache(model_args, "Y")
                if Y is None:
                    print(f"Skipping {model_args}")
                    continue

                print(model_args)
                stamp.set()
                nn = one_NN(X, labels)
                stamp.print_set(f"*\t 1-NN\t {round(nn, 3)}")
                T, C = TC(X, Y, n_neighbors)
                stamp.print(f"*\t C, T\t {round(C, 3)}, {round(T, 3)} ")

                results = [nn, T, C]
                # save_cache(model_args, results, "results")

    input("finished")
