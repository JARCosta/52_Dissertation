import pickle
from matplotlib import pyplot as plt
import numpy as np
import cvxpy as cvx
from sklearn.manifold import Isomap
from sklearn.neighbors import NearestNeighbors

import visualise


def linearise(X_i:np.ndarray, model:str="mvu", k:int=4, d:int=3):
    
    if model.lower() == "isomap":
        isomap = Isomap(n_neighbors=k, n_components=d)
        return isomap.fit_transform(X_i)
    
    elif model.lower() == "mvu":
        nn = NearestNeighbors(n_neighbors=k + 1).fit(X_i)
        nn = nn.kneighbors_graph(X_i).todense()
        nn = np.array(nn) - np.identity(len(nn))

        print(nn)

        connections = []
        for a_idx in range(len(nn)):
            for b_idx in range(len(nn)):
                if nn[a_idx][b_idx] != 0:
                    connections.append([a_idx, b_idx])
        

        print(connections)


        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for a_idx, b_idx in connections:
            a, b = X_i[a_idx], X_i[b_idx]
            ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color="blue", alpha=0.4)
        ax.scatter(X_i[:, 0], X_i[:, 1], X_i[:, 2], color='blue', label='Original Data', s=20, alpha=0.4)
        ax.set_xlabel(f"Component 1")
        ax.set_ylabel(f"Component 2")
        ax.set_zlabel(f"Component 3")
        # ax.set_title(f"{model} of {dataset} Images")
        ax.tick_params(color="white", labelcolor="white")
        plt.show(block=False)



        K = mvu(X_i, nn, cached=True)

        print(K)
        
        eigenvalues, eigenvectors = np.linalg.eig(K)

        # sort the eigenvectors by eigenvalues
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        print("eigenvalues:", eigenvalues[:4])
        
        # plt.plot(eigenvalues)
        # plt.show(block=True)

        print("std:", np.std(eigenvalues))
        print("median:", np.median(eigenvalues))

        eigenvalues_idx = []
        for i in range(len(eigenvalues)):
            if eigenvalues[i] > (np.median(eigenvalues) + np.std(eigenvalues)):
                eigenvalues_idx.append(i)

        # eigenvalues = eigenvalues[eigenvalues_idx]
        # eigenvectors = eigenvectors[:, eigenvalues_idx]

        S_i = np.sqrt(np.abs(eigenvalues))
        S_i = eigenvectors @ np.diag(S_i)

        # print(embeddings.shape)



        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for a_idx, b_idx in connections:
            a, b = S_i[a_idx], S_i[b_idx]
            ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color="blue", alpha=0.4)
        ax.scatter(S_i[:, 0], S_i[:, 1], S_i[:, 2], color='blue', label='Original Data', s=20, alpha=0.4)
        ax.set_xlabel(f"Component 1")
        ax.set_ylabel(f"Component 2")
        ax.set_zlabel(f"Component 3")
        # ax.set_title(f"{model} of {dataset} Images")
        ax.tick_params(color="white", labelcolor="white")
        plt.show(block=True)




        return S_i[:, eigenvalues_idx]
        return S_i


