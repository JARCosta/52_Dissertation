# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tensorflow.keras.datasets import mnist

# Loading the MNIST dataset
(x_train, y_train), (_, _) = mnist.load_data()

# Preprocessing the data: Flattening and normalizing
x_train_flat = x_train.reshape(x_train.shape[0], -1) / 255.0

# Reducing the data to 2D using t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
x_train_tsne = tsne.fit_transform(x_train_flat[:5000])  # Using a subset (5k samples) for faster computation

# Plotting the t-SNE result
plt.figure(figsize=(10, 8))
scatter = plt.scatter(x_train_tsne[:, 0], x_train_tsne[:, 1], c = "blue", s=10, alpha=0.6)
# plt.colorbar(scatter, ticks=range(10))
plt.tick_params(color="white", labelcolor="white")
# plt.title("MNIST Dataset Visualized with t-SNE", fontsize=14)
# plt.xlabel("t-SNE Dimension 1")
# plt.ylabel("t-SNE Dimension 2")
plt.show()
