import numpy as np
from matplotlib import pyplot as plt
import random

def plot_data(ax, data, NM:np.ndarray=np.zeros((1,1)), c="blue", title=""):
    connections = [[a_idx, b_idx] for a_idx in range(NM.shape[0]) for b_idx in range(NM.shape[1]) if NM[a_idx][b_idx] != 0]

    x = data.T[0]
    y = data.T[1] if data.shape[1] > 1 else random.normalvariate(0, 0.001)
    z = data.T[2] if data.shape[1] > 2 else random.normalvariate(0, 0.001)
    ax.scatter(x, y, z, c=c, label='Original Data', s=20, alpha=0.4)
    for a_idx, b_idx in connections:
        a, b = data[a_idx], data[b_idx]
        x_line = [a[0], b[0]]
        y_line = [a[1], b[1]] if data.shape[1] > 1 else [random.normalvariate(0, 0.001), random.normalvariate(0, 0.001)]
        z_line = [a[2], b[2]] if data.shape[1] > 2 else [random.normalvariate(0, 0.001), random.normalvariate(0, 0.001)]
        ax.plot(x_line, y_line, z_line, color="blue", alpha=0.4)
    ax.set_xlabel(f"D1")
    ax.set_ylabel(f"D2")
    ax.set_zlabel(f"D3")
    ax.set_title(title)


def plot_scales(data:np.ndarray, NM:np.ndarray=np.zeros((1,1)), block=True, c="blue", title="", legend=""):
    data = data.real
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={'projection': '3d'})
    fig.text(.1, .1, str(legend))

    plot_data(axs[0], data, NM, c)
    axs[0].set_aspect("equal", adjustable="box")
    
    plot_data(axs[1], data, NM, c)
    
    plt.show(block=block)

def plot_two(data:np.ndarray, data2:np.ndarray, NM:np.ndarray=np.zeros((1,1)), NM2:np.ndarray=np.zeros((1,1)), scale=True, scale2=True, block=True, c="blue", title="", legend=""):
    data, data2 = data.real, data2.real
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={'projection': '3d'})
    fig.text(.1, .1, str(legend))
    
    plot_data(axs[0], data, NM, c)
    axs[0].set_aspect("equal", adjustable="box") if scale else None
    
    plot_data(axs[1], data2, NM2, c)
    axs[1].set_aspect("equal", adjustable="box") if scale2 else None
    
    plt.show(block=block)
    
    

def plot(data:np.ndarray, NM:np.ndarray=np.zeros((1,1)), block=True, c="blue", title="", legend="", scale=True):
    data = data.real
    fig, axs = plt.subplots(1, 1, figsize=(5, 5), subplot_kw={'projection': '3d'})
    fig.text(.1, .1, str(legend))
    plot_data(axs, data, NM, c)
    axs.set_aspect("equal", adjustable="box")
    # axs.tick_params(color="white", labelcolor="white")
    axs.set_title(title)

    plt.show(block=block)
    

def plot_array(data:list[np.ndarray], NM:list[np.ndarray], block=True, c="blue", title="", legend=""):
    from matplotlib import pyplot as plt
    import random

    fig = plt.figure()
    fig.text(.1, .1, str(legend))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(data)):
        data[i] = data[i].real
        connections = [[a_idx, b_idx] for a_idx in range(NM[i].shape[0]) for b_idx in range(NM[i].shape[1]) if NM[i][a_idx][b_idx] != 0]
        x = data[i].T[0]
        y = data[i].T[1] if data[i].shape[1] > 1 else random.normalvariate(0, 0.001)
        z = data[i].T[2] if data[i].shape[1] > 2 else random.normalvariate(0, 0.001)
        ax.scatter(x, y, z, c=c, label='Original Data', s=20, alpha=0.4)
        for a_idx, b_idx in connections:
            a, b = data[i][a_idx], data[i][b_idx]
            x = [a[0], b[0]]
            y = [a[1], b[1]] if data[i].shape[1] > 1 else [random.normalvariate(0, 0.001), random.normalvariate(0, 0.001)]
            z = [a[2], b[2]] if data[i].shape[1] > 2 else [random.normalvariate(0, 0.001), random.normalvariate(0, 0.001)]
            ax.plot(x, y, z, color="blue", alpha=0.4)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"Component 1")
    ax.set_ylabel(f"Component 2")
    ax.set_zlabel(f"Component 3")
    ax.set_title(title)
    # ax.tick_params(color="white", labelcolor="white")
    plt.show(block=block)