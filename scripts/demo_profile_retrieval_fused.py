import json
import numpy as np
import os
import torch
from transformers import AutoTokenizer, AutoModel

# --- CONFIG ---
JSON_PATH = "../LaMP_time_3_subset/train_questions.json"
BGE_PATH = "../bge_emb/task_3_train_bge.memmap.npy"
BGE_OFFSETS_PATH = "../bge_emb/task_3_train_offsets.json"
GRAPH_PATH = "../graph_emb/task_3_graph.npy"
HIS_TO_GRAPH_PATH = "../graph_emb/task_3_his_to_graph.json"
# NODE_MAPS_PATH = "../graph_emb/task_3_node_maps.json"

# --- Load BGE offsets ---
with open(BGE_OFFSETS_PATH, "r") as f:
    bge_offsets = json.load(f)

BGE_MODEL_PATH = "../bge-base-en-v1.5"
TOP_K = 5
ENTRY_IDX = 1
USE_CUSTOM = True  # 🔧 set to True to use custom query text below

CUSTOM_INPUT = """What is the score of the following review on a scale of 1 to 5? 
Just answer with 1, 2, 3, 4, or 5 without further explanation. 
review: It was easy to install, fit perfectly on my bike, and feels very durable. 
The pockets are roomy and well‑made. Great value for money!"""

# Example usage for ENTRY_IDX:
entry_offsets = bge_offsets["entries"][ENTRY_IDX]
profile_id_to_row = {pid: entry_offsets["start"] + i for i, pid in enumerate(entry_offsets.get("profile_id", []))}
# --- LOAD DATA ---
with open(JSON_PATH, "r") as f:
    data = json.load(f)

# --- Load BGE embeddings with correct dtype ---
if BGE_PATH.endswith(".memmap.npy"):
    num_texts = sum(len(entry["profile"]) + 1 for entry in data)
    emb_dim = 768  # Set to your embedding dimension
    emb_bge = np.memmap(BGE_PATH, dtype=np.float32, mode="r", shape=(num_texts, emb_dim))
else:
    emb_bge = np.load(BGE_PATH)

emb_graph = np.load(GRAPH_PATH)
with open(HIS_TO_GRAPH_PATH, "r") as f:
    his_to_graph = {str(k): int(v) for k, v in json.load(f).items()}
# node_maps = json.load(open(NODE_MAPS_PATH))

# --- Select entry and profiles ---
entry = data[ENTRY_IDX]
profile_texts = [his["text"] for his in entry["profile"]]
profile_ids = [str(his["id"]) for his in entry["profile"]]
num_profiles = len(profile_texts)

# --- Compute offset for BGE profile slice ---
offset = 0
for i in range(ENTRY_IDX):
    offset += len(data[i]["profile"]) + 1
profile_embs_bge = emb_bge[offset : offset + num_profiles]
input_emb_bge = emb_bge[offset + num_profiles]

# --- Graph embeddings for profile ---
profile_embs_graph = []
profile_degrees = []
profile_mean_weights = []
GRAPH_CACHE_PATH = "../graph_emb/task_3_graph_cache.pkl"
graph_cache = None
edge_index = None
edge_weight = None

if os.path.exists(GRAPH_CACHE_PATH):
    import pickle
    with open(GRAPH_CACHE_PATH, "rb") as f:
        graph_cache = pickle.load(f)
    edge_index = graph_cache["edge_index"].numpy()
    edge_weight = graph_cache["edge_weight"].numpy()

for hid in profile_ids:
    node_id = his_to_graph.get(hid)
    if node_id is not None and node_id < emb_graph.shape[0]:
        profile_embs_graph.append(emb_graph[int(node_id)])
        # Diagnostics: degree and mean edge weight
        if edge_index is not None and edge_weight is not None:
            neighbors = edge_index[1][edge_index[0] == node_id]
            weights = edge_weight[edge_index[0] == node_id]
            profile_degrees.append(len(neighbors))
            profile_mean_weights.append(float(weights.mean()) if len(weights) > 0 else 0.0)
        else:
            profile_degrees.append(None)
            profile_mean_weights.append(None)
    else:
        profile_embs_graph.append(np.zeros(emb_graph.shape[1], dtype=emb_graph.dtype))
        profile_degrees.append(None)
        profile_mean_weights.append(None)
profile_embs_graph = np.stack(profile_embs_graph) if profile_embs_graph else np.zeros((1, emb_graph.shape[1]))

# --- Utility: embed new text using BGE ---
def embed_bge_text(text: str, model_path: str, device=None):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device or "cpu").eval()
    with torch.no_grad():
        inputs = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device or "cpu")
        out = model(**inputs)
        pooled = (out.last_hidden_state * inputs["attention_mask"].unsqueeze(-1)).sum(1)
        denom = inputs["attention_mask"].sum(1, keepdim=True).clamp(min=1e-6)
        pooled = pooled / denom
        emb = torch.nn.functional.normalize(pooled, p=2, dim=1)
    return emb[0].cpu().numpy()

# --- Input graph embedding selection ---
if USE_CUSTOM:
    input_text = CUSTOM_INPUT.strip()
    print("\n🧠 Embedding custom input text with BGE for graph-only retrieval...")
    input_emb_graph = embed_bge_text(input_text, BGE_MODEL_PATH)
else:
    input_text = entry["input"]
    query_id_str = str(entry["id"])
    node_id = his_to_graph.get(query_id_str)
    if node_id is not None and node_id < emb_graph.shape[0]:
        input_emb_graph = emb_graph[int(node_id)]
        print(f"✅ Found input node in graph: {node_id}")
    else:
        input_emb_graph = np.zeros(emb_graph.shape[1], dtype=emb_graph.dtype)
        print("⚠️ No graph node found for input; input graph embedding is zero.")

# --- Diagnostics ---
print("\nProfile graph embedding norms:")
for i, vec in enumerate(profile_embs_graph):
    print(f"Profile {i}: norm={np.linalg.norm(vec):.3f}")
print(f"Input graph embedding norm: {np.linalg.norm(input_emb_graph):.3f}")

if np.linalg.norm(input_emb_graph) == 0.0:
    print("\n⚠️ Zero embedding for input; similarity scores will all be zero.")

# --- Fuse embeddings ---
def fuse_emb(bge, graph, method="concat", alpha=0.5, normalize=True):
    if method == "concat":
        fused = np.concatenate([bge, graph], axis=-1)
    elif method == "sum":
        fused = alpha * bge + (1 - alpha) * graph
    else:
        raise ValueError("Unknown fusion method")
    if normalize:
        if fused.ndim == 1:
            fused /= np.linalg.norm(fused) + 1e-10
        else:
            norms = np.linalg.norm(fused, axis=1, keepdims=True) + 1e-10
            fused /= norms
    return fused

profile_embs_fused = fuse_emb(profile_embs_bge, profile_embs_graph)
input_emb_fused = fuse_emb(input_emb_bge, input_emb_graph)

# --- Similarity search helpers ---
def show_top_matches(title, sims, texts):
    top_indices = np.argsort(-sims)[:TOP_K]
    print(f"\nTop profile matches for input ({title}):")
    for rank, idx in enumerate(top_indices):
        print(f"Rank {rank+1}: Score={sims[idx]:.3f}")
        print(f"  {texts[idx]}\n")

# --- Retrieval comparisons ---
print(f"\nInput (question+review):\n{input_text}\n")

# --- Compute BGE similarities for each profile ---
sims_bge = np.dot(profile_embs_bge, input_emb_bge)

# --- Compute Graph similarities for each profile ---
sims_graph = np.dot(profile_embs_graph, input_emb_graph)

# --- Print profile texts with diagnostics including graph similarity score ---
print("Profile texts:")
for i, txt in enumerate(profile_texts):
    deg = profile_degrees[i]
    wt = profile_mean_weights[i]
    bge_sim = sims_bge[i]
    graph_sim = sims_graph[i]
    diag_str = ""
    if deg is not None and wt is not None:
        diag_str += f" | degree={deg}, mean_edge_weight={wt:.3f}"
    diag_str += f" | BGE_sim={bge_sim:.3f}"
    diag_str += f" | Graph_Score={graph_sim:.3f}"
    print(f"  [{i}] {txt[:80]}...{diag_str}")

# --- Retrieval comparisons ---
show_top_matches("BGE embeddings only", sims_bge, profile_texts)
print("\n" + "=" * 120 + "\n")

show_top_matches("Graph embeddings only", sims_graph, profile_texts)
print("\n" + "=" * 150 + "\n")

# Fused
sims_fused = np.dot(profile_embs_fused, input_emb_fused)
show_top_matches("fused BGE+Graph embeddings", sims_fused, profile_texts)
print("\n" + "=" * 150 + "\n")

# --- Diagnostics: Graph node degree and edge weights ---
# (Diagnostics now handled during graph embedding construction above)

# --- Similarity statistics ---
def print_similarity_stats(title, sims):
    print(f"\n{title} similarity stats:")
    print(f"  Mean: {np.mean(sims):.3f}")
    print(f"  Std:  {np.std(sims):.3f}")
    print(f"  Max:  {np.max(sims):.3f}")
    print(f"  Min:  {np.min(sims):.3f}")
    print(f"  Top - Mean: {np.max(sims) - np.mean(sims):.3f}")

print_similarity_stats("BGE", sims_bge)
print_similarity_stats("Graph", sims_graph)
print_similarity_stats("Fused", sims_fused)