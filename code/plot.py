import numpy as np


def plot(data:np.ndarray, NM:np.ndarray=np.zeros((1,1)), block=True, c="blue", title="", legend=""):
    from matplotlib import pyplot as plt
    import random
    
    data = data.real

    connections = [[a_idx, b_idx] for a_idx in range(NM.shape[0]) for b_idx in range(NM.shape[1]) if NM[a_idx][b_idx] != 0]

    fig = plt.figure()
    fig.text(.1, .1, str(legend))
    ax = fig.add_subplot(111, projection='3d')
    x = data.T[0]
    y = data.T[1] if data.shape[1] > 1 else random.normalvariate(0, 0.001)
    z = data.T[2] if data.shape[1] > 2 else random.normalvariate(0, 0.001)
    ax.scatter(x, y, z, c=c, label='Original Data', s=20, alpha=0.4)
    for a_idx, b_idx in connections:
        a, b = data[a_idx], data[b_idx]
        x = [a[0], b[0]]
        y = [a[1], b[1]] if data.shape[1] > 1 else [random.normalvariate(0, 0.001), random.normalvariate(0, 0.001)]
        z = [a[2], b[2]] if data.shape[1] > 2 else [random.normalvariate(0, 0.001), random.normalvariate(0, 0.001)]
        ax.plot(x, y, z, color="blue", alpha=0.4)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"Component 1")
    ax.set_ylabel(f"Component 2")
    ax.set_zlabel(f"Component 3")
    ax.set_title(title)
    # ax.tick_params(color="white", labelcolor="white")
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