import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll

from plot import plot

# Generate Swiss Roll dataset (1000 samples, with noise)
X, color = make_swiss_roll(n_samples=1000, noise=0.1, random_state=11)

# Step 1: Compute the Euclidean distance matrix
n = X.shape[0]
distance_matrix = np.zeros((n, n))

plot(X, block=False, c=color)

for i in range(n):
    for j in range(n):
        # diff = X[i] - X[j]   # Compute difference
        # distance_matrix[i, j] = np.sqrt(np.sum(diff**2))  # Squaring, summing, and taking square root
        distance_matrix[i, j] = np.linalg.norm(X[i] - X[j])  # Euclidean distance

# Step 2: Convert the distance matrix into a similarity matrix using an RBF function
gamma = 0.01  # Adjust gamma for different spread levels
similarity_matrix = np.exp(-gamma * distance_matrix**2)  # Gaussian similarity

# Step 3: Center the similarity matrix
H = np.eye(n) - (1/n) * np.ones((n, n))  # Centering matrix
centered_similarity_matrix = H @ similarity_matrix @ H  # Double centering

# Step 4: Eigen decomposition of the similarity matrix
eigenvalues, eigenvectors = np.linalg.eig(similarity_matrix)

# Step 5: Sort eigenvalues and eigenvectors in descending order
eigenvalue_index = np.argsort(eigenvalues)[::-1]
eigenvalues_sorted = eigenvalues[eigenvalue_index]
eigenvectors_sorted = eigenvectors[:, eigenvalue_index]

# Step 6: Project the data onto the top 2 eigenvectors
# top_eigenvectors = eigenvectors_sorted[:, :2]
pca_result = np.dot(similarity_matrix, eigenvectors_sorted)


plot(pca_result, c=color, title="PCA on Euclidean Similarity Matrix - First Two Principal Components")

# # Step 7: Plot the PCA result
# plt.figure(figsize=(8, 6))
# plt.scatter(pca_result[:, 0], pca_result[:, 1], c=color, cmap="viridis", s=5)
# plt.title("PCA on Euclidean Similarity Matrix - First Two Principal Components")
# plt.xlabel("Principal Component 1")
# plt.ylabel("Principal Component 2")
# plt.grid(True)
# plt.show()

# Step 8: Print explained variance ratio
explained_variance_ratio = eigenvalues_sorted / np.sum(eigenvalues_sorted)
print("\nExplained Variance Ratio:")
print(explained_variance_ratio[:5])  # Print first 5 components' variance contribution
