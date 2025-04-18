


from matplotlib import pyplot as plt
import numpy as np
from scipy import linalg


def plot_2D(component_1, component_2, color="blue", block=False):
    plt.figure()
    if color == "blue":
        plt.scatter(component_1, component_2, color='blue', label='Original Data', s=50, alpha=0.6)
    else:
        plt.scatter(component_1, component_2, color='red', label='Selected Points', s=100, edgecolor='black')
    
    plt.xlabel(f" Component 1")
    plt.ylabel(f" Component 2")
    # plt.title(f"{model} of {dataset} Images")
    plt.show(block=block)

def plot_3D(component_1, component_2, component_3, block=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(component_1, component_2, component_3, color='red', label='Selected Points', s=100, edgecolor='black')
    ax.set_xlabel(f"Component 1")
    ax.set_ylabel(f"Component 2")
    ax.set_zlabel(f"Component 3")
    # ax.set_title(f"{model} of {dataset} Images")
    ax.tick_params(color="white", labelcolor="white")
    plt.show(block=block)


def basis(data:np.ndarray):
    cov = ((data - data.mean(0)).T @ data - data.mean(0)) / data.shape[0]
    eigenvalues, eigenvectors = linalg.eigh(cov)
    
    top_eigenvalue_indices = eigenvalues.argsort()[::-1]

    eigenvalues, eigenvectors = eigenvalues[top_eigenvalue_indices], eigenvectors[:, top_eigenvalue_indices]
    
    # for i in range(len(eigenvalues)):
        # print(eigenvalues[i], eigenvectors[i])
    
    print("eigenvalues:", eigenvalues)

    # plt.plot(list(eigenvalues), eigenvalues)
    # plt.show()
    
    return eigenvalues, eigenvectors

def plot(data:np.ndarray=None, highlights:np.ndarray=None, block:bool=True):

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    # ax.tick_params(color="white", labelcolor="white")
    # ax.tick_params(labelcolor="white")


    if np.any(data):
        _, eigenvectors = basis(data)
        data =  data @ eigenvectors @ eigenvectors.T
        
        print(data.shape)

        if len(data.T) >= 3:
            ax.scatter(data[:, 0], data[:, 1], data[:, 2], color='blue', label='Original Data', s=20, alpha=0.4)
        elif len(data.T) == 2:
            ax.scatter(data[:, 0], data[:, 1], 0, color='blue', label='Original Data', s=20, alpha=0.4)
        elif len(data.T) == 1:
            print(data[:, 0])
            ax.scatter(data[:, 0], 1, 2, color='blue', label='Original Data', s=20, alpha=0.4)
    
    elif np.any(highlights):
        _, eigenvectors = basis(highlights)

    if np.any(highlights):
        highlights = highlights @ eigenvectors @ eigenvectors.T
    
        if len(highlights.T) >= 3:
            ax.scatter(highlights[:, 0], highlights[:, 1], highlights[:, 2], color='red', label='Selected Points', s=100, edgecolor='black')
        elif len(highlights.T) == 2:
            ax.scatter(highlights[:, 0], highlights[:, 1], 0, color='red', label='Selected Points', s=100, edgecolor='black')
        elif len(highlights.T) == 1:
            ax.scatter(highlights[:, 0], 0, 0, color='red', label='Selected Points', s=100, edgecolor='black')
    plt.show(block=block)

