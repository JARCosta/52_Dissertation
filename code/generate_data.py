import numpy as np
import sklearn.datasets

from utils import load_cache, save_cache
from plot import plot
import datasets

def generate_data(model_args:dict, noise:float=0.05, random_state:int=None) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:

    if random_state is not None:
        np.random.seed(random_state)

    X, labels, t = None, None, None
    n = model_args['#points']


    if model_args['dataname'] == 'skl.swiss':
        X, t = sklearn.datasets.make_swiss_roll(n_samples=n, noise=noise, random_state=random_state)
        t, height = t.reshape((n, 1)), X[:, 1].reshape((n, 1))

        labels = np.remainder(np.sum(np.hstack([np.round(t / 2), np.round(height / 12)]), axis=1), 2)
        t = np.hstack([t, height])

    elif model_args['dataname'] == 'skl.s_curve':
        X, t = sklearn.datasets.make_s_curve(n_samples=n, noise=noise, random_state=random_state)
        t, height = t.reshape((n, 1)), X[:, 1].reshape((n, 1))

        labels = np.remainder(np.sum(np.hstack([
            np.round(10 * (t / (np.max(t) - np.min(t)))),
            np.round(2 * (height / (np.max(height) - np.min(height))))
        ]), axis=1), 2)
        labels = np.remainder(np.sum(np.hstack([np.round(t), np.round(height)]), axis=1), 2)
        t = np.hstack([t, height])
    
    elif model_args['dataname'] == 'skl.moons':
        X, t = sklearn.datasets.make_moons(n_samples=n, noise=noise, random_state=random_state)
        t = t.reshape((n, 1))
        labels = np.remainder(np.sum(np.round(t), axis=1), 2)
    
        return X, labels, t


    elif model_args['dataname'] == 'swiss':
        return datasets.swiss.default(n, noise, random_state=random_state)

    elif model_args['dataname'] == 's_curve':
        return datasets.s_curve.default(n, noise, random_state=random_state)


    elif model_args['dataname'] == 'broken.swiss':
        return datasets.swiss.broken(n, noise, random_state=random_state)

    elif model_args['dataname'] == 'broken.s_curve':
        return datasets.s_curve.broken(n, noise, random_state=random_state)


    elif model_args['dataname'] == 'paralell.swiss':
        return datasets.parallel(n, noise, random_state=random_state)

    elif model_args['dataname'] == 'changing.swiss':
        return datasets.swiss.changing(n, noise, random_state=random_state)


    elif model_args['dataname'] == 'toro.swiss':
        return datasets.swiss.toro(n, noise, random_state=random_state)

    elif model_args['dataname'] == 'four.moons':
        return datasets.moons.four(n, noise, random_state=random_state)

    elif model_args['dataname'] == 'helix':
        t = np.linspace(1, n, n)[:, None] / n
        t = (t ** 1.0) * 2 * np.pi
        X = np.hstack([(2 + np.cos(8 * t)) * np.cos(t), (2 + np.cos(8 * t)) * np.sin(t), np.sin(8 * t)]) + noise * np.random.randn(n, 3)
        labels = np.remainder(np.round(t * 1.5), 2)


    elif model_args['dataname'] == 'twinpeaks':
        inc = 1.5 / np.sqrt(n)
        xx, yy = np.meshgrid(np.arange(-1, 1 + inc, inc), np.arange(-1, 1 + inc, inc))
        xy = 1 - 2 * np.random.rand(2, n)
        X = np.hstack([xy.T, (np.sin(np.pi * xy[0, :]) * np.tanh(3 * xy[1, :]))[:, None]]) + noise * np.random.randn(n, 3)
        X[:, 2] *= 10
        labels = np.remainder(np.sum(np.round((X - np.min(X, axis=0)) / 10), axis=1), 2)


    elif model_args['dataname'] == '3d_clusters':
        num_clusters = 5
        centers = 10 * np.random.rand(num_clusters, 3)
        D = np.linalg.norm(centers[:, None] - centers[None, :], axis=2)
        min_distance = np.min(D[D > 0])
        X = np.zeros((n, 3))
        labels = np.zeros(n, dtype=int)
        k = 0
        n2 = n - (num_clusters - 1) * 9
        for i in range(num_clusters):
            for _ in range(int(np.ceil(n2 / num_clusters))):
                if k < n:
                    X[k] = centers[i] + (np.random.rand(3) - 0.5) * min_distance / np.sqrt(12)
                    labels[k] = i + 1
                    k += 1
        X += noise * np.random.randn(n, 3)


    elif model_args['dataname'] == 'intersect':
        t = np.linspace(1, n, n)[:, None] / n * (2 * np.pi)
        x = np.cos(t)
        y = np.sin(t)
        height = np.random.rand(len(x), 1) * 5
        X = np.hstack([x, x * y, height]) + noise * np.random.randn(n, 3)
        labels = np.remainder(np.sum(np.hstack([np.round(t / 2), np.round(height / 2)]), axis=1), 2)


    elif model_args['dataname'] == 'difficult':
        no_dims = 5
        no_points_per_dim = round(n ** (1 / no_dims))
        l = np.linspace(0, 1, no_points_per_dim)
        t = np.array(np.meshgrid(*[l] * no_dims)).T.reshape(-1, no_dims)
        X = np.hstack([np.cos(t[:, 0])[:, None], np.tanh(3 * t[:, 1])[:, None], (t[:, 0] + t[:, 2])[:, None], 
                       (t[:, 3] * np.sin(t[:, 1]))[:, None], (np.sin(t[:, 0] + t[:, 4]))[:, None], 
                       (t[:, 4] * np.cos(t[:, 1]))[:, None], (t[:, 4] + t[:, 3])[:, None], 
                       t[:, 1][:, None], (t[:, 2] * t[:, 3])[:, None], t[:, 0][:, None]])
        X += noise * np.random.randn(*X.shape)
        labels = np.remainder(np.sum(1 + np.round(t), axis=1), 2)


    else:
        raise ValueError(f"Unknown dataset name {model_args['dataname']}.")

    # save_cache(model_args, X, "X")
    # save_cache(model_args, labels, "l")
    # if type(t) != type(None):
    #     save_cache(model_args, t, "t")
    return X, labels, t

def import_data(model_args:dict) -> tuple[np.ndarray, np.ndarray, None]:

    X, labels, t = None, None, None

    if model_args['dataname'] == "teapots":
        from scipy.io import loadmat
        X = loadmat('datasets/teapots.mat')
        X = X["Input"][0][0][0]
        X = X.T.astype(np.float64)
        X = X - X.mean(0)
    elif model_args['dataname'] == "mnist":
        def read_images_labels(images_filepath, labels_filepath):
            import struct
            from array import array

            labels = []
            with open(labels_filepath, 'rb') as file:
                magic, size = struct.unpack(">II", file.read(8))
                if magic != 2049:
                    raise ValueError(f"Magic number mismatch, expected 2049, got {magic}")
                labels = array("B", file.read())
            
            with open(images_filepath, 'rb') as file:
                magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
                if magic != 2051:
                    raise ValueError(f"Magic number mismatch, expected 2051, got {magic}")
                image_data = array("B", file.read())
            images = []
            for i in range(size):
                images.append([0] * rows * cols)
            for i in range(size):
                img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
                img = img.reshape(28, 28)
                images[i][:] = img
            
            images = np.array(images)

            return images.reshape(images.shape[0], -1), np.array(labels)
        training_images_filepath = 'datasets/mnist/train-images.idx3-ubyte'
        training_labels_filepath = 'datasets/mnist/train-labels.idx1-ubyte'
        test_images_filepath = 'datasets/mnist/t10k-images.idx3-ubyte'
        test_labels_filepath = 'datasets/mnist/t10k-labels.idx1-ubyte'
        x_train, y_train = read_images_labels(training_images_filepath, training_labels_filepath)
        x_test, y_test = read_images_labels(test_images_filepath, test_labels_filepath)
        print(x_train.shape, y_train.shape)
        print(x_test.shape, y_test.shape)
        
        # X = np.vstack([x_train, x_test])
        # labels = np.vstack([y_train, y_test])
        X, labels = x_train, y_train

        print(X.shape, labels.shape)
    
    elif model_args['dataname'] == "coil20":
        from PIL import Image
        import os
        
        X = []
        labels = []
        for i in range(1,21):
            data_directory = f"datasets/coil20/{i}/"
            file_names = [n for n in os.listdir(data_directory) if n[-3:] == "png"]
            file_names.sort()
            
            for file_name in file_names:
                image = Image.open(data_directory + file_name)
                x = np.array(image)
                X.append(x.reshape(x.shape[0] * x.shape[1]))
                labels.append(i)

        X, labels = np.vstack(X), np.vstack(labels)

        print("X:", X.shape)
        print("labels:", labels.shape)

    elif model_args['dataname'] == "nisis":
        raise ValueError(f"TODO: import dataset {model_args['dataname']}.")

    elif model_args['dataname'] == "orl":
        from PIL import Image
        import os
        
        X = []
        labels = []
        for i in range(1,41):
            data_directory = f"datasets/orl/s{i}/"
            file_names = [n for n in os.listdir(data_directory) if n[-3:] == "pgm"]
            file_names.sort()
            
            for file_name in file_names:
                image = Image.open(data_directory + file_name)
                x = np.array(image)
                X.append(x.reshape(x.shape[0] * x.shape[1]))
                labels.append(np.array([i, ]))

        print(type(X), type(labels))
        X, labels = np.vstack(X), np.vstack(np.array(labels))
        print(type(X), type(labels))

        print(X, X.shape)
        print(labels, labels.shape)
    
    elif model_args['dataname'] == "hiva":
        raise ValueError(f"TODO: import dataset {model_args['dataname']}.")

    else:
        raise ValueError(f"Unknown dataset name {model_args['dataname']}.")
    
    
    # save_cache(model_args, X, "X")
    # if type(labels) != type(None):
    #     save_cache(model_args, labels, "l")
    # if type(t) != type(None):
    #     save_cache(model_args, t, "t")

    return X, labels, t












def load_set(model_args:dict) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    X = load_cache(model_args, "X")
    labels = load_cache(model_args, "l")
    t = load_cache(model_args, "t")

    return X, labels, t

def get_dataset(model_args:dict, cache:bool=True, random_state:int=None) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:

    if cache:
        X, labels, t = load_set(model_args)
        if type(X) != type(None):
            from utils import bcolors
            print(bcolors.LIGHTGREEN + "Dataset loaded from cache" + bcolors.ENDC)
            return X, labels, t

    try:
        return generate_data(model_args, random_state=random_state)
    except ValueError as e:
        print(f"Error generating dataset: {e}")
    try:
        return import_data(model_args)
    except ValueError:
        raise ValueError(f"Can't load or generate dataset {model_args['dataname']}")
    