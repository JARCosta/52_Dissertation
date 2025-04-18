import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll

# Generate Swiss Roll dataset (1000 samples, with random noise)
X, color = make_swiss_roll(n_samples=1000, noise=0.1)
data_matrix = X

print("Original Data:")
print(data_matrix)

# Step 1: Standardize the data (mean = 0, variance = 1)
mean = np.mean(data_matrix, axis=0)
std_dev = np.std(data_matrix, axis=0)
data_standardized = (data_matrix - mean) / std_dev

print(data_standardized)
# Step 2: Calculate the covariance matrix
cov_matrix = np.cov(data_standardized.T)

# Step 3: Compute eigenvalues and eigenvectors using np.linalg.eig()
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Step 4: Sort the eigenvalues and corresponding eigenvectors in descending order
eigenvalue_index = np.argsort(eigenvalues)[::-1]
eigenvalues_sorted = eigenvalues[eigenvalue_index]
eigenvectors_sorted = eigenvectors[:, eigenvalue_index]

# Step 5: Project the data onto the principal components (select top 2 components)
top_eigenvectors = eigenvectors_sorted[:, :2]  # Select top 2 eigenvectors
pca_result = np.dot(data_standardized, top_eigenvectors)

# Convert the PCA result to a DataFrame
pca_df = pd.DataFrame(pca_result, columns=['Principal Component 1', 'Principal Component 2'])

# Output the results
print("\nPCA Result (Top 2 Components):")
print(pca_df)

# Explained Variance (proportion of variance explained by each component)
explained_variance_ratio = eigenvalues_sorted / np.sum(eigenvalues_sorted)
print("\nExplained Variance Ratio:")
print(explained_variance_ratio)

# Visualizing the PCA result
plt.figure(figsize=(8, 6))
plt.scatter(pca_df['Principal Component 1'], pca_df['Principal Component 2'], color='blue')
plt.title('PCA - First Two Principal Components')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()
