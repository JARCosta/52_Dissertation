import numpy as np
from PIL import Image
import os

def mnist():

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
    training_images_filepath = 'code/datasets/mnist/train-images.idx3-ubyte'
    training_labels_filepath = 'code/datasets/mnist/train-labels.idx1-ubyte'
    test_images_filepath = 'code/datasets/mnist/t10k-images.idx3-ubyte'
    test_labels_filepath = 'code/datasets/mnist/t10k-labels.idx1-ubyte'
    x_train, y_train = read_images_labels(training_images_filepath, training_labels_filepath)
    # x_test, y_test = read_images_labels(test_images_filepath, test_labels_filepath)
    # print(x_train.shape, y_train.shape)
    # print(x_test.shape, y_test.shape)
    
    # X = np.vstack([x_train, x_test])
    # labels = np.vstack([y_train, y_test])
    X, labels = x_train, y_train

    # print(X.shape, labels.shape)
    
    return X, labels, None

def coil20():
    
    X = []
    labels = []
    for i in range(1,21):
        data_directory = f"code/datasets/coil20/{i}/"
        file_names = [n for n in os.listdir(data_directory) if n[-3:] == "png"]
        file_names.sort()
        
        for file_name in file_names:
            image = Image.open(data_directory + file_name)
            x = np.array(image)
            X.append(x.reshape(x.shape[0] * x.shape[1]))
            labels.append(i)

    X, labels = np.vstack(X), np.vstack(labels)
    X = X.astype(np.float64)

    # print("X:", X.shape)
    # print("labels:", labels.shape)

    return X, labels, None


def orl():

    
    X = []
    labels = []
    for i in range(1,41):
        data_directory = f"code/datasets/orl/s{i}/"
        file_names = [n for n in os.listdir(data_directory) if n[-3:] == "pgm"]
        file_names.sort()
        
        for file_name in file_names:
            image = Image.open(data_directory + file_name)
            x = np.array(image)
            X.append(x.reshape(x.shape[0] * x.shape[1]))
            labels.append(np.array([i, ]))

    X, labels = np.vstack(X), np.vstack(np.array(labels))
    X = X.astype(np.float64)

    # print(X, X.shape)
    # print(labels, labels.shape)

    return X, labels, None


def hiva(version:str='prior'):
    from rdkit import Chem
    
    base_path = f"code/datasets/HIVA-{version}"
    
    # Read training data
    train_file = os.path.join(base_path, "hiva_train.sd" if version == "prior" else "hiva_train.data")
    train_labels_file = os.path.join(base_path, "hiva_train.labels")
    
    # Read molecules
    suppl = Chem.SDMolSupplier(train_file)
    X = []
    for mol in suppl:
        if mol is not None:
            # Convert molecule to fingerprint
            fp = Chem.RDKFingerprint(mol)
            X.append(np.array(fp))
    
    X = np.vstack(X)
    
    # Read labels
    labels = np.loadtxt(train_labels_file, dtype=int)
    
    X = X.astype(np.float64)
    
    print(f"* HIVA-{version} dataset loaded ({X.shape[0]} points).")
    # stamp.print(f"* HIVA-{version} dataset loaded ({X.shape[0]} points).")

    return X, labels, None

def mit_cbcl():
    """
    Load MIT-CBCL face recognition database.
    
    Returns:
        X: numpy array of flattened images
        labels: numpy array of subject labels (0-9)
        None: no additional metadata
    """
    X = []
    labels = []
    
    # Target size for all images (64x64 is common for face datasets)
    target_size = (64, 64)
    
    # Load training images (JPG format)
    training_dir = "code/datasets/MIT-CBCL-facerec-database/training-originals/"
    if os.path.exists(training_dir):
        for filename in os.listdir(training_dir):
            if filename.endswith('.jpg'):
                # Extract subject ID from filename (e.g., "0001_00000001.jpg" -> 1)
                subject_id = int(filename.split('_')[0])
                
                # Load and process image
                image_path = os.path.join(training_dir, filename)
                image = Image.open(image_path)
                
                # Convert to grayscale if needed and resize
                if image.mode != 'L':
                    image = image.convert('L')
                image = image.resize(target_size, Image.Resampling.LANCZOS)
                
                x = np.array(image)
                
                # Flatten the image
                X.append(x.reshape(-1))
                labels.append(subject_id)
    
    # Load test images (PGM format)
    test_dir = "code/datasets/MIT-CBCL-facerec-database/test/"
    if os.path.exists(test_dir):
        for filename in os.listdir(test_dir):
            if filename.endswith('.pgm'):
                # Extract subject ID from filename (e.g., "0009_06042.pgm" -> 9)
                subject_id = int(filename.split('_')[0])
                
                # Load and process image
                image_path = os.path.join(test_dir, filename)
                image = Image.open(image_path)
                
                # Convert to grayscale if needed and resize
                if image.mode != 'L':
                    image = image.convert('L')
                image = image.resize(target_size, Image.Resampling.LANCZOS)
                
                x = np.array(image)
                
                # Flatten the image
                X.append(x.reshape(-1))
                labels.append(subject_id)
    
    if not X:
        raise ValueError("No images found in MIT-CBCL dataset directories")
    
    # Convert to numpy arrays
    X = np.array(X, dtype=np.float64)
    labels = np.array(labels).reshape(-1, 1)
    
    print(f"MIT-CBCL dataset loaded: {X.shape[0]} images, {X.shape[1]} features")
    print(f"Number of subjects: {len(np.unique(labels))}")
    print(f"Image size: {target_size[0]}x{target_size[1]} = {X.shape[1]} features")
    
    return X, labels, None

def teapots():
    from scipy.io import loadmat
    
    X = loadmat('code/datasets/teapots.mat')
    X = X["Input"][0][0][0]
    X = X.T.astype(np.float64)
    
    return X, None, None