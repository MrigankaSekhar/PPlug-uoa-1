
import os
import json
import time
import argparse
import torch
import gc
import numpy as np
import itertools
from typing import Optional, Dict, Tuple
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import sys, os

# -----------------------------
# CONFIG
# -----------------------------

USE_SUBSET = True

GRAPHSAGE_EPOCHS = 50
GRAPHSAGE_HIDDEN_DIM = 256
TASK_ID = 3

DEV_DATASET_FOLDER_SUBSET = f"LaMP_time_{TASK_ID}_subset"
FULL_DATASET_FOLDER = f"LaMP_time_{TASK_ID}"


GRAPH_DIR = "../graph_emb"
os.makedirs(GRAPH_DIR, exist_ok=True)
SAVE_PATH = os.path.join(GRAPH_DIR, f"task_{TASK_ID}_graph.npy")
MAP_PATH = os.path.join(GRAPH_DIR, f"task_{TASK_ID}_his_to_graph.json")
# ✅ Pick dataset paths based on subset/full config
if USE_SUBSET:
    TRAIN_FILE = os.path.join("..", DEV_DATASET_FOLDER_SUBSET, "train_questions.json")
    DEV_FILE   = os.path.join("..", DEV_DATASET_FOLDER_SUBSET, "dev_questions.json")
else:
    TRAIN_FILE = os.path.join("..", FULL_DATASET_FOLDER, "train_questions.json")
    DEV_FILE   = os.path.join("..", FULL_DATASET_FOLDER, "dev_questions.json")

print(f"📄 TRAIN_FILE = {TRAIN_FILE}")
print(f"📄 DEV_FILE   = {DEV_FILE}")

SCHEMA = [
    ("User", "Item", "RATED"),
    ("User", "Review", "WROTE"),
    ("Review", "Item", "DESCRIBES"),
    ("Item", "Category", "BELONGS_TO"),
]

# Relation weights (Option 1)
REL_WEIGHT = {
    "RATED":        1.00,
    "WROTE":        1.20,
    "DESCRIBES":    0.80,
    "BELONGS_TO":   0.60,
}

# Field map (if you later add more)
FIELD_MAP = {
    "User":     "user_id",
    "Item":     "item_id",
    "Review":   "review_text",   # We will override from 'input' and 'profile[*].text'
    "Category": "category",
}

FEATURE_INIT = "bge"
BGE_MODEL_PATH = "../bge-base-en-v1.5/"
EMB_DIM = 768
BGE_OUTPUT_DIM = 768

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

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
    """Extract review text from the 'input' field if it contains 'review:'."""
    if not isinstance(inp, str):
        return ""
    low = inp.lower()
    k = low.find("review:")
    return inp[k+len("review:"):].strip() if k >= 0 else inp.strip()

# -----------------------------
# BUILD GRAPH (global IDs)
# -----------------------------
# One global ID space to avoid collisions and memmap gaps
node_index: Dict[Tuple[str, str], int] = {}    # (type, key) -> global_id
node_maps: Dict[str, Dict[str, int]] = {t: {} for t, _, _ in SCHEMA} | {t: {} for _, t, _ in SCHEMA}
node_key_texts: Dict[str, Dict[str, str]] = {t: {} for t, _, _ in SCHEMA} | {t: {} for _, t, _ in SCHEMA}
next_id = 0

edges = []       # list[(src_id, tgt_id)]
edge_wts = []    # list[float] (relation-aware weights)

his_to_graph = {}  # maps profile 'id' (string) -> global review node id

def add_node(typ: str, key: str, text: Optional[str] = None) -> int:
    """Create or fetch a global node ID; store text if provided."""
    global next_id
    if (typ, key) not in node_index:
        node_index[(typ, key)] = next_id
        node_maps[typ][key] = next_id
        next_id += 1
    if isinstance(text, str) and text.strip():
        node_key_texts[typ][key] = text.strip()
    return node_index[(typ, key)]

def add_edge(src_id: int, tgt_id: int, rel: str):
    """Append undirected edges with relation weight."""
    w = REL_WEIGHT.get(rel, 1.0)
    edges.append((src_id, tgt_id)); edge_wts.append(w)
    edges.append((tgt_id, src_id)); edge_wts.append(w)



def process_entry(entry: dict, from_train: bool = True):
    # Question-level nodes
    user_key = entry.get("user_id") or f"user_{entry.get('id')}"
    user_text = f"User {user_key}"

    # Review text from 'input' — only embed for train entries to avoid test leakage
    review_text_q = ""
    if from_train:
        review_text_q = _extract_review_from_input(entry.get("input", ""))
    else:
        # 🔍 Debug message for leakage check
        print(f"⚠️ Skipping question review text for dev entry id={entry.get('id')}")

    review_key_q = f"review_{entry.get('id')}"

    # Synthetic item (no item_title present)
    item_key_q = f"item_{entry.get('id')}"
    item_text_q = f"Item for question {entry.get('id')}"

    user_id = add_node("User", user_key, user_text)

    # Only include review text if from train set
    review_id_q = add_node("Review", review_key_q, review_text_q if from_train else None)
    item_id_q = add_node("Item", item_key_q, item_text_q)

    # Wire edges (respect SCHEMA)
    add_edge(user_id, review_id_q, "WROTE")
    add_edge(review_id_q, item_id_q, "DESCRIBES")
    add_edge(user_id, item_id_q, "RATED")

    # Category only if present
    cat = entry.get("category")
    if cat:
        cat_id = add_node("Category", str(cat), str(cat))
        add_edge(item_id_q, cat_id, "BELONGS_TO")

    # Profile reviews (history)
    if "profile" in entry and isinstance(entry["profile"], list):
        for his in entry["profile"]:
            hid = str(his.get("id"))
            his_text = his.get("text", "")
            p_review_id = add_node("Review", hid, his_text)
            add_edge(user_id, p_review_id, "WROTE")
            his_to_graph[hid] = p_review_id

            # Synthetic item per profile review (optional)
            p_item_key = f"item_{hid}"
            p_item_id = add_node("Item", p_item_key, f"Item (profile {hid})")
            add_edge(p_review_id, p_item_id, "DESCRIBES")
            add_edge(user_id, p_item_id, "RATED")

# Load & slice data
train_data = load_json_lines(TRAIN_FILE)
dev_data = load_json_lines(DEV_FILE)

# if USE_SUBSET and SUBSET_SIZE_JSON and SUBSET_SIZE_JSON > 0:
#     print(f"⚠️ Using subset mode for JSON data: first {SUBSET_SIZE_JSON} entries from train & dev")
#     train_data = train_data[:SUBSET_SIZE_JSON]
#     dev_data   = dev_data[:SUBSET_SIZE_JSON]

# for entry in itertools.chain(train_data, dev_data):
#     process_entry(entry)

# -----------------------------
# SUBSET FILTERING (limit nodes)
# -----------------------------
# if USE_SUBSET and SUBSET_SIZE_GRAPH_EMBED_NODES and SUBSET_SIZE_GRAPH_EMBED_NODES > 0:
#     print(f"⚡ SUBSET MODE: reducing graph to first {SUBSET_SIZE_GRAPH_EMBED_NODES} nodes")
#     # Allowed nodes are the first N by global id (deterministic)
#     all_ids_sorted = sorted(node_index.values())
#     allowed_nodes = set(all_ids_sorted[:SUBSET_SIZE_GRAPH_EMBED_NODES])

#     # Filter per-type maps
#     for typ in list(node_maps.keys()):
#         node_maps[typ] = {k: v for k, v in node_maps[typ].items() if v in allowed_nodes}

#     # Filter node_key_texts at key level
#     for typ in list(node_key_texts.keys()):
#         node_key_texts[typ] = {k: t for k, t in node_key_texts[typ].items()
#                                if node_maps[typ].get(k) in allowed_nodes}

#     # Filter edges + weights together
#     new_edges, new_wts = [], []
#     for (s, t), w in zip(edges, edge_wts):
#         if s in allowed_nodes and t in allowed_nodes:
#             new_edges.append((s, t)); new_wts.append(w)
#     edges, edge_wts = new_edges, new_wts

#     # Filter his_to_graph
#     his_to_graph = {k: v for k, v in his_to_graph.items() if v in allowed_nodes}


# -----------------------------
# Process train entries (include question review text)
# -----------------------------
for entry in train_data:
    process_entry(entry, from_train=True)

# -----------------------------
# Process dev entries (skip question review text to avoid leakage)
# -----------------------------
for entry in dev_data:
    process_entry(entry, from_train=False)

# -----------------------------
# ID COMPACTION (critical for dev)
# -----------------------------
# Compact remaining IDs to 0..K-1 before creating any memmaps
remaining_ids = set()
for m in node_maps.values():
    remaining_ids.update(m.values())
for s, t in edges:
    remaining_ids.add(s); remaining_ids.add(t)

id_list_sorted = sorted(remaining_ids)
old2new = {old: new for new, old in enumerate(id_list_sorted)}
num_nodes = len(id_list_sorted)  # compact count

# Remap node_maps (per-type)
for typ in list(node_maps.keys()):
    node_maps[typ] = {k: old2new[v] for k, v in node_maps[typ].items() if v in old2new}

# Build id->text from node_key_texts (new IDs)
node_texts = {}
for typ, mapping in node_maps.items():
    for key, new_id in mapping.items():
        txt = node_key_texts[typ].get(key)
        if isinstance(txt, str) and txt.strip():
            node_texts[new_id] = txt.strip()

# Remap edges (order preserved so weights align)
edges = [(old2new[s], old2new[t]) for (s, t) in edges if s in old2new and t in old2new]
# edge_wts already aligned with edges during prior filtering; order unchanged

# Remap his_to_graph
his_to_graph = {k: old2new[v] for k, v in his_to_graph.items() if v in old2new}

# Build tensors from compact IDs
edge_index_list = [(s, t) for (s, t) in edges]
edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
edge_weight = torch.tensor(edge_wts, dtype=torch.float)

print(f"📊 COMPACT graph: {num_nodes} nodes; edges={len(edges)}")

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

    tokenizer = AutoTokenizer.from_pretrained(BGE_MODEL_PATH)
    model = AutoModel.from_pretrained(BGE_MODEL_PATH, torch_dtype=torch.float32).to(device).eval()

    # Nodes that actually have text
    text_nodes_all = sorted([(txt, gid) for gid, txt in node_texts.items()], key=lambda p: p[1])

    #Optional: further limit the number of text nodes embedded
    # if USE_SUBSET and SUBSET_SIZE_GRAPH_EMBED_NODES and SUBSET_SIZE_GRAPH_EMBED_NODES > 0:
    #     text_nodes_all = text_nodes_all[:SUBSET_SIZE_GRAPH_EMBED_NODES]
    #     print(f"⚠️ Limiting embedding stage to {len(text_nodes_all)} text nodes")

    # Chunk slicing
    chunk_size = max(1, len(text_nodes_all) // max(1, num_chunks))
    start_i = chunk_index * chunk_size
    end_i = (chunk_index + 1) * chunk_size if (chunk_index < num_chunks - 1) else len(text_nodes_all)
    text_nodes = [tn for tn in text_nodes_all[start_i:end_i] if tn[1] not in processed_ids]

    batch_size = 256 if device.type != 'cuda' else 2048
    save_every = 50 if device.type != 'cuda' else 200

    print(f"📦 Embedding chunk {chunk_index+1}/{num_chunks}: {len(text_nodes)} nodes, batch size {batch_size}")

    for batch_num, i in enumerate(tqdm(range(0, len(text_nodes), batch_size), desc="Embedding", unit="batch")):
        batch_segment = text_nodes[i:i+batch_size]
        inputs = tokenizer([node[0] for node in batch_segment],
                           return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
        with torch.no_grad():
            out = model(**inputs)
            pooled = _mean_pool(out.last_hidden_state, inputs["attention_mask"])
            embeddings = torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy().astype(np.float16)

        for j, idx in enumerate([node[1] for node in batch_segment]):
            x[idx] = embeddings[j]
            processed_ids.add(idx)

        if ((batch_num + 1) % save_every == 0) or (i + batch_size >= len(text_nodes)):
            x.flush()
            json.dump(list(processed_ids), open(PROCESSED_IDS_PATH, "w"))

    print(f"✅ Done embedding chunk in {time.time()-start_time_all:.2f}s, total processed {len(processed_ids)}")
    x.flush()
    json.dump(list(processed_ids), open(PROCESSED_IDS_PATH, "w"))

    if device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()

def merge_chunks_to_full():
    print("🔄 Merging partial embeddings...")
    x_partial = np.memmap(PARTIAL_X_PATH, dtype=np.float16, mode='r', shape=(num_nodes, EMB_DIM))
    x_final   = np.memmap(X_SAVE_PATH, dtype=np.float16, mode='w+', shape=(num_nodes, EMB_DIM))
    for i in tqdm(range(num_nodes), desc="Merging", unit="row"):
        x_final[i] = x_partial[i]
    x_final.flush()
    print(f"✅ Merged to {X_SAVE_PATH}")
    try:
        os.remove(PARTIAL_X_PATH)
        print(f"🗑️ Deleted partial file: {PARTIAL_X_PATH}")
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