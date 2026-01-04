
import os
import json
import time
import argparse
import torch
import gc
import numpy as np
from typing import Optional, Dict, Tuple
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import sys
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
try:
    from nltk.sentiment import SentimentIntensityAnalyzer
except LookupError:
    nltk.download('vader_lexicon')
    from nltk.sentiment import SentimentIntensityAnalyzer
# -----------------------------
# CONFIG
# -----------------------------
USE_SUBSET = True
GRAPHSAGE_EPOCHS = 10
GRAPHSAGE_HIDDEN_DIM = 256
TASK_ID = 3

DEV_DATASET_FOLDER_SUBSET = f"LaMP_time_{TASK_ID}_subset"
FULL_DATASET_FOLDER = f"LaMP_time_{TASK_ID}"

GRAPH_DIR = "../graph_emb"
os.makedirs(GRAPH_DIR, exist_ok=True)
SAVE_PATH = os.path.join(GRAPH_DIR, f"task_{TASK_ID}_graph.npy")
MAP_PATH = os.path.join(GRAPH_DIR, f"task_{TASK_ID}_his_to_graph.json")

if USE_SUBSET:
    TRAIN_FILE = os.path.join("..", DEV_DATASET_FOLDER_SUBSET, "train_questions.json")
    DEV_FILE   = os.path.join("..", DEV_DATASET_FOLDER_SUBSET, "dev_questions.json")
else:
    TRAIN_FILE = os.path.join("..", FULL_DATASET_FOLDER, "train_questions.json")
    DEV_FILE   = os.path.join("..", FULL_DATASET_FOLDER, "dev_questions.json")

print(f"📄 TRAIN_FILE = {TRAIN_FILE}")
print(f"📄 DEV_FILE   = {DEV_FILE}")

# -----------------------------
# NEW SCHEMA with Sentiment & Popularity
# -----------------------------
SCHEMA = [
    ("User", "Item", "RATED"),
    ("User", "Review", "WROTE"),
    ("Review", "Item", "DESCRIBES"),
    ("Review", "Sentiment", "HAS_POLARITY"),
    ("Item", "Popularity", "HAS_POP"),
]

REL_WEIGHT = {
    "RATED":        1.00,
    "WROTE":        1.20,
    "DESCRIBES":    0.80,
    "HAS_POLARITY": 0.70,
    "HAS_POP":      0.50,
}

FIELD_MAP = {
    "User":     "user_id",
    "Item":     "item_id",
    "Review":   "review_text",
    "Sentiment": "sentiment",
    "Popularity": "popularity_level",
}

FEATURE_INIT = "bge"
BGE_MODEL_PATH = "../bge-base-en-v1.5/"
EMB_DIM = 768
BGE_OUTPUT_DIM = 768

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Sentiment analyzer (rule-based, works offline)
sia = SentimentIntensityAnalyzer()

# -----------------------------
# UTILS
# -----------------------------
def load_json_lines(path):
    with open(path) as f:
        try:
            return [json.loads(line) for line in f]
        except json.JSONDecodeError:
            f.seek(0)
            return json.load(f)

def _extract_review_from_input(inp: str) -> str:
    if not isinstance(inp, str): return ""
    low = inp.lower()
    k = low.find("review:")
    return inp[k+len("review:"):].strip() if k >= 0 else inp.strip()

def text_to_sentiment(text: str) -> Optional[str]:
    """Derive sentiment bucket from review text using VADER compound score."""
    if not isinstance(text, str) or not text.strip():
        return None
    score = sia.polarity_scores(text)['compound']
    if score >= 0.2:
        return "Positive"
    elif score <= -0.2:
        return "Negative"
    else:
        return "Neutral"

# -----------------------------
# GRAPH BUILD
# -----------------------------
node_index: Dict[Tuple[str, str], int] = {}
node_maps: Dict[str, Dict[str, int]] = {t: {} for t, _, _ in SCHEMA} | {t: {} for _, t, _ in SCHEMA}
node_key_texts: Dict[str, Dict[str, str]] = {t: {} for t, _, _ in SCHEMA} | {t: {} for _, t, _ in SCHEMA}
next_id = 0
edges = []
edge_wts = []
his_to_graph = {}
item_freq = {}  # For popularity binning

def add_node(typ: str, key: str, text: Optional[str] = None) -> int:
    global next_id
    if (typ, key) not in node_index:
        node_index[(typ, key)] = next_id
        node_maps[typ][key] = next_id
        next_id += 1
    if isinstance(text, str) and text.strip():
        node_key_texts[typ][key] = text.strip()
    return node_index[(typ, key)]

def add_edge(src_id: int, tgt_id: int, rel: str):
    w = REL_WEIGHT.get(rel, 1.0)
    edges.append((src_id, tgt_id)); edge_wts.append(w)
    edges.append((tgt_id, src_id)); edge_wts.append(w)

def get_popularity_bucket(item_key: str) -> str:
    """Map item frequency to a coarse bucket."""
    freq = item_freq.get(item_key, 0)
    if freq >= 10:
        return "HighPop"
    elif freq >= 3:
        return "MedPop"
    else:
        return "LowPop"

def process_entry(entry: dict, from_train: bool = True):
    # User node
    user_key = entry.get("user_id") or f"user_{entry.get('id')}"
    user_id = add_node("User", user_key, f"User {user_key}")

    # Review node for the query
    review_key_q = f"review_{entry.get('id')}"
    review_text_q = _extract_review_from_input(entry.get("input", "")) if from_train else None
    review_id_q = add_node("Review", review_key_q, review_text_q if from_train else None)

    # Item node (synthetic)
    item_key_q = f"item_{entry.get('id')}"
    item_id_q = add_node("Item", item_key_q, f"Item for question {entry.get('id')}")

    # Count for popularity
    item_freq[item_key_q] = item_freq.get(item_key_q, 0) + 1

    # Core edges
    add_edge(user_id, review_id_q, "WROTE")
    add_edge(review_id_q, item_id_q, "DESCRIBES")
    add_edge(user_id, item_id_q, "RATED")

    # Sentiment edge for query
    if review_text_q:
        bucket = text_to_sentiment(review_text_q)
        if bucket:
            sent_id = add_node("Sentiment", bucket, bucket)
            add_edge(review_id_q, sent_id, "HAS_POLARITY")

    # Popularity edge for query
    pop_bucket = get_popularity_bucket(item_key_q)
    pop_id = add_node("Popularity", pop_bucket, pop_bucket)
    add_edge(item_id_q, pop_id, "HAS_POP")

    # Profile history reviews
    if "profile" in entry and isinstance(entry["profile"], list):
        for his in entry["profile"]:
            hid = str(his.get("id"))
            his_text = his.get("text", "")
            p_review_id = add_node("Review", hid, his_text)
            add_edge(user_id, p_review_id, "WROTE")
            his_to_graph[hid] = p_review_id

            # Link profile review to sentiment
            bucket = text_to_sentiment(his_text)
            if bucket:
                sent_id = add_node("Sentiment", bucket, bucket)
                add_edge(p_review_id, sent_id, "HAS_POLARITY")

            # Synthetic item for profile review
            p_item_key = f"item_{hid}"
            p_item_id = add_node("Item", p_item_key, f"Item (profile {hid})")
            add_edge(p_review_id, p_item_id, "DESCRIBES")
            add_edge(user_id, p_item_id, "RATED")

            # Popularity edge
            item_freq[p_item_key] = item_freq.get(p_item_key, 0) + 1
            pop_bucket = get_popularity_bucket(p_item_key)
            pop_id = add_node("Popularity", pop_bucket, pop_bucket)
            add_edge(p_item_id, pop_id, "HAS_POP")

# -----------------------------
# Load data
# -----------------------------
train_data = load_json_lines(TRAIN_FILE)
dev_data = load_json_lines(DEV_FILE)

# First pass to count item frequencies
print("📊 Counting item frequencies...")
for entry in tqdm(train_data + dev_data, desc="Counting items", unit="entry"):
    # synthetic item ID matches how process_entry will name it
    item_key_q = f"item_{entry.get('id')}"
    item_freq[item_key_q] = item_freq.get(item_key_q, 0) + 1
    if "profile" in entry:
        for his in entry["profile"]:
            p_item_key = f"item_{str(his.get('id'))}"
            item_freq[p_item_key] = item_freq.get(p_item_key, 0) + 1

# Second pass to build nodes/edges
print("🏗 Building graph from TRAIN set...")
for entry in tqdm(train_data, desc="Processing train_data", unit="entry"):
    process_entry(entry, from_train=True)

print("🏗 Building graph from DEV set...")
for entry in tqdm(dev_data, desc="Processing dev_data", unit="entry"):
    process_entry(entry, from_train=False)

# -----------------------------
# COMPACT IDs and build tensors (same logic as original)
# -----------------------------
remaining_ids = set()
for m in node_maps.values():
    remaining_ids.update(m.values())
for s, t in edges:
    remaining_ids.add(s); remaining_ids.add(t)

old2new = {old: new for new, old in enumerate(sorted(remaining_ids))}
num_nodes = len(old2new)

for typ in list(node_maps.keys()):
    node_maps[typ] = {k: old2new[v] for k, v in node_maps[typ].items() if v in old2new}

node_texts = {}
for typ, mapping in node_maps.items():
    for key, new_id in mapping.items():
        txt = node_key_texts[typ].get(key)
        if isinstance(txt, str) and txt.strip():
            node_texts[new_id] = txt.strip()

edges = [(old2new[s], old2new[t]) for (s, t) in edges]
his_to_graph = {k: old2new[v] for k, v in his_to_graph.items() if v in old2new}

edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
edge_weight = torch.tensor(edge_wts, dtype=torch.float)

print(f"📊 COMPACT graph: {num_nodes} nodes; edges={len(edges)}")

# === Rest of embed/train/infer functions stay exactly as your original ===

# -----------------------------
# PATHS
# -----------------------------
X_SAVE_PATH = os.path.join(GRAPH_DIR, f"task_{TASK_ID}_x.npy")
PARTIAL_X_PATH = os.path.join(GRAPH_DIR, f"task_{TASK_ID}_x_partial.npy")
PROCESSED_IDS_PATH = os.path.join(GRAPH_DIR, f"task_{TASK_ID}_processed_ids.json")
MODEL_CKPT_PATH = os.path.join(GRAPH_DIR, "graphsage_epoch{}.pt")

# -----------------------------
# EMBEDDING
# -----------------------------
def _mean_pool(last_hidden_state, attention_mask):
    """Attention-mask aware mean pooling (recommended for sentence embeddings)."""
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return summed / denom


def embed_chunk(chunk_index, num_chunks):
    """
    Embeds graph nodes that have associated text using the BGE model.
    Reuses precomputed his_embed vectors for Review nodes via his_to_graph.
    """
    start_time_all = time.time()

    if os.path.exists(PARTIAL_X_PATH):
        print(f"♻️ Resuming from partial NP checkpoint: {PARTIAL_X_PATH}")
        x = np.memmap(PARTIAL_X_PATH, dtype=np.float16, mode='r+',
                      shape=(num_nodes, EMB_DIM))
        processed_ids = set(json.load(open(PROCESSED_IDS_PATH))) if os.path.exists(PROCESSED_IDS_PATH) else set()
    else:
        print("🚀 Starting fresh NP embedding for this chunk.")
        x = np.memmap(PARTIAL_X_PATH, dtype=np.float16, mode='w+',
                      shape=(num_nodes, EMB_DIM))
        processed_ids = set()

    # --- 1️⃣ Load precomputed his_embed table ---
    # Adjust to your actual his_embed path (should match profile emb table file)
    his_embed_path = "../bge_emb/task_3_dev_bge.npy"
    if os.path.exists(his_embed_path):
        if his_embed_path.endswith(".npy"):
            his_embed_table = np.load(his_embed_path)
        else:
            his_embed_table = torch.load(his_embed_path, map_location="cpu").numpy()
        print(f"📂 Loaded his_embed table from {his_embed_path} shape={his_embed_table.shape}")
        # Populate Review nodes directly
        assign_count = 0
        for his_id_str, node_id in his_to_graph.items():
            his_id = int(his_id_str)
            if his_id < his_embed_table.shape[0]:
                x[node_id] = his_embed_table[his_id].astype(np.float16)
                processed_ids.add(node_id)
                assign_count += 1
        print(f"✅ Assigned {assign_count} Review node embeddings from precomputed his_embed")
    else:
        print(f"⚠️ his_embed_path '{his_embed_path}' not found — falling back to BGE for all nodes.")

    # --- 2️⃣ Continue with BGE for the rest of node_texts (non‑Review) ---
    tokenizer = AutoTokenizer.from_pretrained(BGE_MODEL_PATH)
    model = AutoModel.from_pretrained(BGE_MODEL_PATH, torch_dtype=torch.float32).to(device).eval()

    data_obj = Data(edge_index=edge_index, num_nodes=num_nodes)
    text_nodes_all = sorted([(txt, gid) for gid, txt in node_texts.items() if gid not in processed_ids],
                            key=lambda p: p[1])

    if not text_nodes_all:
        print("✅ No new nodes to process in this chunk (all covered by his_embed or processed).")
        x.flush()
        json.dump(list(processed_ids), open(PROCESSED_IDS_PATH, "w"))
        return

    chunk_size = max(1, len(text_nodes_all) // max(1, num_chunks))
    start_i = chunk_index * chunk_size
    end_i = (chunk_index + 1) * chunk_size if (chunk_index < num_chunks - 1) else len(text_nodes_all)
    target_text_nodes = [tn for tn in text_nodes_all[start_i:end_i]]

    loader = NeighborLoader(
        data_obj,
        input_nodes=torch.tensor([gid for _, gid in target_text_nodes], dtype=torch.long),
        num_neighbors=[0],
        batch_size=2048 if device.type == 'cuda' else 256,
        shuffle=False
    )

    save_every = 10 if device.type != 'cuda' else 50
    total_processed = 0

    for batch_num, batch in enumerate(tqdm(loader, desc="Embedding text nodes", unit="batch")):
        texts = [node_texts[n.item()] for n in batch.n_id if n.item() in node_texts and n.item() not in processed_ids]
        if not texts:
            continue

        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
        with torch.no_grad():
            out = model(**inputs)
            pooled = _mean_pool(out.last_hidden_state, inputs["attention_mask"])
            embeddings = torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy().astype(np.float16)

        idxs = [n.item() for n in batch.n_id if n.item() in node_texts and n.item() not in processed_ids]
        for j, node_id in enumerate(idxs):
            x[node_id] = embeddings[j]
            processed_ids.add(node_id)
            total_processed += 1

        if (batch_num + 1) % save_every == 0:
            x.flush()
            json.dump(list(processed_ids), open(PROCESSED_IDS_PATH, "w"))

    x.flush()
    json.dump(list(processed_ids), open(PROCESSED_IDS_PATH, "w"))
    print(f"✅ Done embedding chunk in {time.time() - start_time_all:.2f}s, total processed={total_processed + assign_count}")

    if device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
def merge_chunks_to_full():
    """Merge partial embedding memmap into the final X_SAVE_PATH file.
       Also fills any missing Review node embeddings from precomputed his_embed."""
    if not os.path.exists(PARTIAL_X_PATH):
        raise FileNotFoundError(f"❌ Partial embedding file not found: {PARTIAL_X_PATH}")

    # Map the partial file in read mode
    try:
        x_partial = np.memmap(PARTIAL_X_PATH, dtype=np.float16, mode='r', shape=(num_nodes, EMB_DIM))
    except Exception as e:
        raise RuntimeError(f"❌ Failed to open partial memmap: {e}")

    if x_partial.shape != (num_nodes, EMB_DIM):
        raise ValueError(f"❌ Shape mismatch: expected {(num_nodes, EMB_DIM)}, got {x_partial.shape}")

    # --- 🔍 Fill in any missing review node vectors from his_embed ---
    his_embed_path = "../bge_emb/task_3_dev_bge.emb"
    if os.path.exists(his_embed_path):
        his_embed_table = torch.load(his_embed_path, map_location="cpu").numpy()
        zero_rows = 0
        for his_id_str, node_id in his_to_graph.items():
            if np.allclose(x_partial[node_id], 0, atol=1e-8):
                his_id = int(his_id_str)
                if his_id < his_embed_table.shape[0]:
                    x_partial[node_id] = his_embed_table[his_id].astype(np.float16)
                    zero_rows += 1
        if zero_rows > 0:
            print(f"🛠 Filled {zero_rows} missing Review node vectors from his_embed in merge step")
    else:
        print(f"⚠️ his_embed_path '{his_embed_path}' not found — cannot backfill missing Review nodes at merge.")

    print(f"🔄 Merging partial embeddings ({x_partial.shape[0]} nodes, dim={x_partial.shape[1]}) → {X_SAVE_PATH}")

    # Allocate the final memmap and copy in one vectorised operation
    x_final = np.memmap(X_SAVE_PATH, dtype=np.float16, mode='w+', shape=(num_nodes, EMB_DIM))
    x_final[:] = x_partial[:]
    x_final.flush()

    print(f"✅ Merged to {X_SAVE_PATH} (dtype={x_final.dtype}, size={x_final.nbytes/1e6:.2f} MB)")

    # Clean up partial
    try:
        os.remove(PARTIAL_X_PATH)
        if os.path.exists(PROCESSED_IDS_PATH):
            os.remove(PROCESSED_IDS_PATH)
        print(f"🗑️ Deleted partial file(s): {PARTIAL_X_PATH} and processed IDs list")
    except OSError as e:
        print(f"⚠️ Could not delete partial file {PARTIAL_X_PATH}: {e}")
# -----------------------------
# TRAIN
# -----------------------------
def train_gnn():
    print("🚀 Training GNN...")

    # Load base embeddings (float32 for stability)
    x_array = np.memmap(X_SAVE_PATH, dtype=np.float16, mode='r', shape=(num_nodes, EMB_DIM))
    x_tensor = torch.tensor(x_array.astype(np.float32))

    # Build Data with edge_attr as weights
    data_obj = Data(x=x_tensor, edge_index=edge_index, edge_attr=edge_weight).to(device)

    ALPHA = 0.2  # residual weight (keep close to original)

    class GraphSAGE(torch.nn.Module):
        def __init__(self, in_c, h_c, out_c, alpha):
            super().__init__()
            self.alpha = alpha
            self.conv1 = SAGEConv(in_c, h_c)
            self.conv2 = SAGEConv(h_c, out_c)
            self.dropout = nn.Dropout(0.3)

        def forward(self, x, edge_index, edge_weight=None):
            # Pass edge_weight if supported
            try:
                h = self.conv1(x, edge_index, edge_weight=edge_weight).relu()
            except TypeError:
                h = self.conv1(x, edge_index).relu()
            h = self.dropout(h)
            try:
                h = self.conv2(h, edge_index, edge_weight=edge_weight)
            except TypeError:
                h = self.conv2(h, edge_index)
            out = self.alpha * h + (1 - self.alpha) * x
            return torch.nn.functional.normalize(out, p=2, dim=1)

    gnn_model = GraphSAGE(EMB_DIM, GRAPHSAGE_HIDDEN_DIM, EMB_DIM, alpha=ALPHA).to(device)
    optimizer = torch.optim.AdamW(gnn_model.parameters(), lr=1e-3, weight_decay=1e-4)

    bs = 1024 if USE_SUBSET else 512
    neigh = [10, 5] if USE_SUBSET else [15, 10]
    loader = NeighborLoader(
        data_obj,
        num_neighbors=neigh,
        batch_size=bs,
        shuffle=True
    )

    for epoch in range(GRAPHSAGE_EPOCHS):
        gnn_model.train()
        total_loss = 0.0

        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{GRAPHSAGE_EPOCHS}", unit="batch"):
            optimizer.zero_grad()

            pred = gnn_model(batch.x, batch.edge_index, edge_weight=getattr(batch, 'edge_attr', None))
            target = torch.nn.functional.normalize(batch.x, p=2, dim=1)

            # mask out nodes with near-zero originals
            mask = (batch.x.norm(dim=1) > 1e-6)
            if mask.sum() == 0:
                continue

            # consistency to original
            cos_loss = 1 - torch.nn.functional.cosine_similarity(pred[mask], target[mask]).mean()

            # relation-weighted neighbor smoothing on sampled edges
            src, dst = batch.edge_index
            nb_cos = torch.nn.functional.cosine_similarity(pred[src], pred[dst])
            if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
                w = batch.edge_attr
                nb_loss = 1 - ( (w * nb_cos).sum() / w.sum().clamp(min=1e-9) )
            else:
                nb_loss = 1 - nb_cos.mean()

            loss = 0.7 * cos_loss + 0.3 * nb_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gnn_model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        print(f"📉 Epoch {epoch+1} loss={total_loss/len(loader):.4f}")
        torch.save(gnn_model.state_dict(), MODEL_CKPT_PATH.format(epoch+1))

# -----------------------------
# INFER
# -----------------------------
# ... all your existing imports and code above remain unchanged ...

def infer_gnn():
    print("🚀 Inferring with GNN...")

    x_array = np.memmap(X_SAVE_PATH, dtype=np.float16, mode='r', shape=(num_nodes, EMB_DIM))
    x_tensor = torch.tensor(x_array.astype(np.float32))
    data_obj = Data(x=x_tensor, edge_index=edge_index, edge_attr=edge_weight).to(device)

    ALPHA = 0.02  # must match training

    class GraphSAGE(torch.nn.Module):
        def __init__(self, in_c, h_c, out_c, alpha):
            super().__init__()
            self.alpha = alpha
            self.conv1 = SAGEConv(in_c, h_c)
            self.conv2 = SAGEConv(h_c, out_c)

        def forward(self, x, edge_index, edge_weight=None):
            try:
                h = self.conv1(x, edge_index, edge_weight=edge_weight).relu()
            except TypeError:
                h = self.conv1(x, edge_index).relu()
            try:
                h = self.conv2(h, edge_index, edge_weight=edge_weight)
            except TypeError:
                h = self.conv2(h, edge_index)
            out = self.alpha * h + (1 - self.alpha) * x
            return torch.nn.functional.normalize(out, p=2, dim=1)

    # Load the latest checkpoint
    ckpts = sorted([f for f in os.listdir(GRAPH_DIR) if f.startswith("graphsage_epoch")],
                   key=lambda f: int(f.split("epoch")[1].split(".")[0]))
    latest_ckpt = ckpts[-1]
    gnn_model = GraphSAGE(EMB_DIM, GRAPHSAGE_HIDDEN_DIM, EMB_DIM, alpha=ALPHA).to(device)
    gnn_model.load_state_dict(torch.load(os.path.join(GRAPH_DIR, latest_ckpt), map_location=device))
    gnn_model.eval()

    # We'll accumulate results in RAM first (safe for your subset sizes)
    final_array = np.zeros((num_nodes, EMB_DIM), dtype=np.float16)

    bs = 4096 if USE_SUBSET else 256
    neigh = [-1] if USE_SUBSET else [-1]
    loader = NeighborLoader(data_obj, num_neighbors=neigh, batch_size=bs, shuffle=False)

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Final inference", unit="batch"):
            out = gnn_model(batch.x, batch.edge_index, edge_weight=getattr(batch, 'edge_attr', None))
            final_array[batch.n_id] = out.cpu().numpy().astype(np.float16)

    # Save his_to_graph mapping
    json.dump(his_to_graph, open(MAP_PATH, "w"))

    # Save a proper .npy file with header
    try:
        np.save(SAVE_PATH, final_array)
        print(f"💾 Saved embeddings to proper NumPy .npy format at {SAVE_PATH}")
    except Exception as e:
        print(f"❌ Failed to save proper .npy format: {e}")

    print(f"✅ {num_nodes} normalized embeddings ready.")

# -----------------------------
# METRICS
# -----------------------------
def compute_metrics_only():
    print("📈 Computing metrics...")

    if os.path.exists(PROCESSED_IDS_PATH):
        eval_ids = sorted(set(json.load(open(PROCESSED_IDS_PATH))))
        print(f"⚠️ Subset metrics: {len(eval_ids)} embedded nodes")
    else:
        eval_ids = list(range(num_nodes))
        print(f"ℹ️ Full metrics: all {len(eval_ids)} nodes")

    # Load graph file (.npy format) properly
    final_mm = np.load(SAVE_PATH, mmap_mode='r').astype(np.float32)
    # Load base embeddings (raw memmap) properly
    orig_mm  = np.memmap(X_SAVE_PATH, dtype=np.float16, mode='r', shape=(num_nodes, EMB_DIM)).astype(np.float32)

    final_sel = final_mm[eval_ids]
    orig_sel  = orig_mm[eval_ids]

    final_t = torch.nn.functional.normalize(torch.tensor(final_sel), p=2, dim=1).numpy()
    orig_t  = torch.nn.functional.normalize(torch.tensor(orig_sel),  p=2, dim=1).numpy()

    zero_count = np.sum(np.linalg.norm(final_t, axis=1) == 0)
    if zero_count > 0:
        print(f"⚠️ {zero_count} / {len(eval_ids)} evaluated nodes have zero embeddings")
    else:
        print("✅ All evaluated nodes have non‑zero embeddings")

    chunk_size = 20000
    mse_accum = 0.0
    cos_accum = 0.0

    for start in tqdm(range(0, len(eval_ids), chunk_size), desc="MSE/Cosine", unit="chunk"):
        fe = final_t[start:start+chunk_size]
        ox = orig_t[start:start+chunk_size]

        mse_accum += torch.nn.functional.mse_loss(torch.tensor(fe), torch.tensor(ox), reduction='sum').item()
        num = np.sum(fe * ox, axis=1)
        den = np.linalg.norm(fe, axis=1) * np.linalg.norm(ox, axis=1)
        cos_accum += np.sum(num / np.clip(den, 1e-9, None))

    print(f" • MSE:    {mse_accum / len(eval_ids):.6f}")
    print(f" • Cosine: {cos_accum / len(eval_ids):.6f}")
# -----------------------------
# CLI ENTRY
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["embed","merge","train","infer","metrics"])
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--num-chunks", type=int, default=1)
    args = parser.parse_args()

    if args.stage == "embed":
        embed_chunk(args.chunk_index, args.num_chunks)
    elif args.stage == "merge":
        merge_chunks_to_full()
    elif args.stage == "train":
        train_gnn()
    elif args.stage == "infer":
        infer_gnn()
    elif args.stage == "metrics":
        compute_metrics_only()