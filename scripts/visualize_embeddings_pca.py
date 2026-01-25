import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import json
import os

# Get shape and dtype from your config or metadata
with open("../graph_emb/task_3_num_nodes.json") as f:
    num_nodes = int(json.load(f))
emb_dim = 768  # or whatever your embedding dimension is

# --- Select embedding file (prefer raw) ---
raw_path = "../graph_emb/task_3_x_raw.npy"
norm_path = "../graph_emb/task_3_x.npy"
emb_path = raw_path if os.path.exists(raw_path) else norm_path

# Load embeddings (use float32 for safety)
embeddings = np.memmap(emb_path, dtype=np.float32, mode="r", shape=(num_nodes, emb_dim))
print(f"Using embeddings from: {emb_path}")
print("Shape:", embeddings.shape)

# Load node type mapping (assumes node_maps file exists)
node_maps_path = "../graph_emb/task_3_node_maps.json"
with open(node_maps_path, "r") as f:
    node_maps = json.load(f)

# Build a node_id -> type mapping
node_type = ["Unknown"] * embeddings.shape[0]
for typ, id_map in node_maps.items():
    for k, v in id_map.items():
        if v < len(node_type):
            node_type[v] = typ

# Assign a color for each type
type_to_color = {
    "User": "blue",
    "Review": "green",
    "Item": "orange",
    "Sentiment": "red",
    "Popularity": "purple",
    "Category": "brown",
    "Unknown": "gray",
}
colors = [type_to_color.get(t, "gray") for t in node_type]

# --- Run PCA on normalized, centered data ---
print("Normalizing and centering embeddings for PCA...")
emb_arr = np.array(embeddings, dtype=np.float32)

# Normalize each embedding vector to unit length to avoid overflow
norms = np.linalg.norm(emb_arr, axis=1, keepdims=True)
emb_arr = emb_arr / np.clip(norms, 1e-9, None)

# Center embeddings
emb_arr -= emb_arr.mean(axis=0)

# Run PCA (on smaller sample if dataset huge)
if emb_arr.shape[0] > 20000:
    print("Large dataset detected — subsampling for PCA...")
    sample_idx = np.random.choice(len(emb_arr), size=10000, replace=False)
    emb_arr_pca = emb_arr[sample_idx]
else:
    emb_arr_pca = emb_arr

pca = PCA(n_components=2)
emb_2d = pca.fit_transform(emb_arr_pca)
print("Explained variance:", pca.explained_variance_ratio_)

# --- Plot, color by node type ---
plt.figure(figsize=(10, 8))
for typ in set(node_type):
    idxs = [i for i, t in enumerate(node_type[: len(emb_arr_pca)]) if t == typ]
    plt.scatter(
        emb_2d[idxs, 0],
        emb_2d[idxs, 1],
        s=6,
        alpha=0.5,
        label=typ,
        c=type_to_color.get(typ, "gray"),
    )

plt.title("PCA of Raw Graph Embeddings (colored by node type)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(markerscale=2)
plt.tight_layout()
plt.show()