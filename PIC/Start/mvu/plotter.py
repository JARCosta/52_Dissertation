
import numpy as np
import matplotlib.pyplot as plt

def save_gram(Gram, Gram_path):
    with open(Gram_path + ".txt", 'w') as f:
        for i in Gram:
            f.write(str(i).replace("\n", "") + '\n')
    np.save(Gram_path + ".npy", Gram)

def save_eigenvalues(eigenvalues, eigenvalues_path):
    with open(eigenvalues_path + ".txt", 'w') as f:
        f.write(str(eigenvalues).replace("\n", "") + '\n')
    # np.save(eigenvalues_path + ".npy", eigenvalues)

def load_gram(gram_path):
    Gram = np.load(gram_path)
    return Gram

def plot_2D(component_1, component_2, model, dataset):
    plt.scatter(component_1, component_2)
    plt.xlabel(f"{model} Component 1")
    plt.ylabel(f"{model} Component 2")
    plt.title(f"{model} of {dataset} Images")
    plt.show(block=False)

def plot_3D(component_1, component_2, component_3, model, dataset, block=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(component_1, component_2, component_3)
    ax.set_xlabel(f"{model} Component 1")
    ax.set_ylabel(f"{model} Component 2")
    ax.set_zlabel(f"{model} Component 3")
    ax.set_title(f"{model} of {dataset} Images")
    plt.show(block=block)

def plot_check_ortonormality(eigenvectors):
    ortonormality = []
    for i in range(len(eigenvectors)):
        for j in range(len(eigenvectors)):
            if i != j:
                ortonormality.append(np.log10(abs(eigenvectors[i] @ eigenvectors[j])))
    plt.hist(ortonormality, bins=100, range=(min(ortonormality), 0))
    plt.show()

def plot_Gram(Gram, dataname, block=True):
    abs = np.abs(Gram)
    
    # plot 1x2 Gram matrix and its absolute value and color bars
    fig, axs = plt.subplots(1, 2, figsize=(15,5))
    axs[0].imshow(Gram)
    axs[0].set_title(f"Inner Product between Points of {dataname}")
    axs[1].imshow(abs)
    axs[1].set_title(f"Absolute Value of Inner Product between Points of {dataname}")
    plt.colorbar(axs[0].imshow(Gram), ax=axs[0])
    plt.colorbar(axs[1].imshow(abs), ax=axs[1])
    plt.show(block=block)

def plot_MVU(Gram, dataname, block=True):

    eigenvalues, eigenvectors = np.linalg.eig(Gram)

    # sort the eigenvectors by eigenvalues
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    values = np.sqrt(np.abs(eigenvalues))
    values = eigenvectors @ np.diag(values)

    print("eigenvalues:\n",eigenvalues)
    save_eigenvalues(eigenvalues, f"resources/{dataname}.eigenvalues")
    
    plot_check_ortonormality(eigenvectors)

    plot_2D(values[:, 0], values[:, 1], 'MVU', dataname)
    plot_3D(values[:, 0], values[:, 1], values[:, 2], 'MVU', dataname, block)

def plot_connections(connections, n_neighbors):

    neighbors = []
    for i in range(n_neighbors):
        neighbors.append([])

    for node in range(connections.shape[0]):
        # print(node, np.where(connections[node] == 1)[0], np.where(connections[node] == 1)[0]/node)
        # print(list(np.where(connections[node] == 1)[0]))
        for i in range(n_neighbors):
            neighbors[i].append(sorted(list(np.where(connections[node] == 1)[0]))[i])

    # print(len(range(connections.shape[0])), len(neighbors))

    for i in range(connections.shape[0]):
        t_neighbors = [neighbors[j][i] for j in range(n_neighbors)]

        higher, lower = 0, 0

        for nei in t_neighbors:
            if i < nei:
                higher += 1
            else:
                lower += 1
        
        if higher == 0 or lower == 0:
            print(i, t_neighbors, higher, lower)



    # plt.plot(nodes , neighbors1)
    # plt.show()

    # plt.plot(nodes , neighbors2)
    # plt.show()
    
    # plt.plot(nodes , neighbors3)
    # plt.show()

def save_problem(objective, constraints, filename):
    with open(filename, 'w') as f:
        f.write(f"{objective}\n")
        for constraint in constraints:
            f.write(f"{constraint}\n")
        f.write("Length of constraints: " + str(len(constraints)))

if __name__ == '__main__':
    gram_path = 'resources/saves/mvu.teapots.5e-04.3.gram.npy'
    Gram = load_gram(gram_path)
    plot_MVU(Gram, gram_path.split('.')[0])
    plot_Gram(Gram, gram_path.split('.')[0])