import hashlib
import pickle
import numpy as np
import os

class DimensionalityReductionCache:
    def __init__(self, cache_dir="cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _hash_params(self, X, **params):
        """Create a unique hash for the dataset and parameters."""
        data_hash = hashlib.sha256(X.tobytes()).hexdigest()
        params_hash = hashlib.sha256(str(sorted(params.items())).encode()).hexdigest()
        return f"{data_hash}_{params_hash}"

    def save(self, X, embedding, **params):
        """Save the computed embedding in the cache."""
        key = self._hash_params(X, **params)
        cache_path = os.path.join(self.cache_dir, key + ".pkl")
        
        with open(cache_path, "wb") as f:
            pickle.dump({"X": X, "embedding": embedding, "params": params}, f)

    def load(self, X, **params):
        """Retrieve an existing result from the cache, or return the best approximation."""
        data_hash = hashlib.sha256(X.tobytes()).hexdigest()
        best_match = None
        best_eps = float("inf")
        best_n_neighbors = 0
        best_n_components = 0
        
        for filename in os.listdir(self.cache_dir):
            if filename.startswith(data_hash):
                with open(os.path.join(self.cache_dir, filename), "rb") as f:
                    cached = pickle.load(f)
                
                cached_params = cached["params"]
                
                # Exact match
                if cached_params == params:
                    return cached["embedding"]
                
                # For MVU: Return the result with the smallest available `eps` that is <= requested `eps`
                if "eps" in params and "eps" in cached_params:
                    if cached_params["eps"] <= params["eps"] and cached_params["eps"] < best_eps:
                        best_match = cached["embedding"]
                        best_eps = cached_params["eps"]
                
                # For Isomap: Return the result with the largest `n_neighbors` or `n_components` ≤ requested
                if "n_neighbors" in params and "n_neighbors" in cached_params:
                    if cached_params["n_neighbors"] <= params["n_neighbors"] and cached_params["n_neighbors"] > best_n_neighbors:
                        best_match = cached["embedding"]
                        best_n_neighbors = cached_params["n_neighbors"]

                if "n_components" in params and "n_components" in cached_params:
                    if cached_params["n_components"] <= params["n_components"] and cached_params["n_components"] > best_n_components:
                        best_match = cached["embedding"]
                        best_n_components = cached_params["n_components"]
        
        return best_match

# Example usage
if __name__ == "__main__":
    from sklearn.manifold import Isomap
    from sklearn.datasets import make_swiss_roll

    # Generate a dataset
    X, _ = make_swiss_roll(n_samples=1000, noise=0.1)

    # Create cache
    cache = DimensionalityReductionCache()

    # Run Isomap with specific parameters
    iso = Isomap(n_neighbors=10, n_components=2)
    embedding = iso.fit_transform(X)

    # Save the solution
    cache.save(X, embedding, n_neighbors=10, n_components=2)

    # Retrieve it later
    retrieved_embedding = cache.load(X, n_neighbors=10, n_components=2)
    print("Cache Hit:", retrieved_embedding is not None)

    # Try retrieving with a smaller `eps` or a different `n_neighbors`
    retrieved_embedding = cache.load(X, n_neighbors=8, n_components=2)
    print("Approximate Match Found:", retrieved_embedding is not None)
