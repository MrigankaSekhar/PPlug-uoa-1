import numpy as np
import os

X_PATH = "../graph_emb/task_3_x.npy"

# If both np.load (with and without allow_pickle) fail with _pickle.UnpicklingError,
# your file is likely NOT a valid .npy file, but a memory-mapped array created with np.memmap,
# or it is corrupted or not a NumPy file at all.

# Try loading as a memory-mapped array directly:
try:
    # Try to infer the shape and dtype (most likely float16, 2D)
    # If you know the shape and dtype, set them explicitly!
    # Example: shape=(num_nodes, 768), dtype=np.float16
    # You may need to check your graph construction code for the correct shape.
    shape = None
    dtype = None

    # Try to infer shape/dtype from a cache or from the file size
    # For demonstration, let's try (N, 768) and float16
    # If you have a cache file with num_nodes, use it!
    CACHE_PATH = "../graph_emb/task_3_graph_cache.pkl"
    if os.path.exists(CACHE_PATH):
        import pickle
        with open(CACHE_PATH, "rb") as f:
            cache = pickle.load(f)
        num_nodes = cache.get("num_nodes")
        dtype = np.float16
        shape = (num_nodes, 768)
    else:
        # Fallback: try to infer from file size
        file_size = os.path.getsize(X_PATH)
        dtype = np.float16
        dim = 768
        num_nodes = file_size // (np.dtype(dtype).itemsize * dim)
        shape = (num_nodes, dim)

    x = np.memmap(X_PATH, dtype=dtype, mode="r", shape=shape)
    print(f"Loaded memmap with shape {x.shape} and dtype {x.dtype}")
except Exception as e:
    print(f"Memmap load failed: {e}")
    raise RuntimeError("File could not be loaded as .npy or memmap. Check file format and creation code.")

num_total = x.shape[0]
num_nonzero = int((np.abs(x).sum(axis=1) > 0).sum())

print(f"Total nodes: {num_total}")
print(f"Nodes with embeddings: {num_nonzero}")
print(f"Percent complete: {num_nonzero/num_total:.2%}")