import os
import json
import time
import argparse
import torch
import gc
import numpy as np
import pickle
from typing import Optional, Dict, Tuple
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import sys
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer

# -----------------------------
# CONFIG
# -----------------------------
USE_SUBSET = True
GRAPHSAGE_EPOCHS = 50
GRAPHSAGE_HIDDEN_DIM = 512
TASK_ID = 3

DEV_DATASET_FOLDER_SUBSET = f"LaMP_time_{TASK_ID}_subset"
FULL_DATASET_FOLDER = f"LaMP_time_{TASK_ID}"

GRAPH_DIR = "../graph_emb"
os.makedirs(GRAPH_DIR, exist_ok=True)
CACHE_PATH = os.path.join(GRAPH_DIR, f"task_{TASK_ID}_graph_cache.pkl")
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

FEATURE_INIT = "bge"
BGE_MODEL_PATH = "../bge-base-en-v1.5/"
EMB_DIM = 768

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

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
    if not isinstance(text, str) or not text.strip():
        return None
    score = sia.polarity_scores(text)['compound']
    if score >= 0.2:  return "Positive"
    elif score <= -0.2: return "Negative"
    else: return "Neutral"

# -----------------------------
# GRAPH BUILD
# -----------------------------
def build_graph_and_cache():
    print("🏗 Building graph once and caching...")

    SCHEMA = [
        ("User", "Item", "RATED"),
        ("User", "Review", "WROTE"),
        ("Review", "Item", "DESCRIBES"),
        ("Review", "Sentiment", "HAS_POLARITY"),
        ("Item", "Popularity", "HAS_POP"),
    ]
    REL_WEIGHT = {
        "RATED":        1.0,
        "WROTE":        1.2,
        "DESCRIBES":    0.8,
        "HAS_POLARITY": 0.7,
        "HAS_POP":      0.5,
    }
    node_index = {}
    node_maps = {t: {} for t, _, _ in SCHEMA} | {t: {} for _, t, _ in SCHEMA}
    node_texts = {t: {} for t, _, _ in SCHEMA} | {t: {} for _, t, _ in SCHEMA}
    next_id = 0
    edges, edge_wts = [], []
    his_to_graph, item_freq = {}, {}

    def add_node(typ, key, text=None):
        nonlocal next_id
        if (typ, key) not in node_index:
            node_index[(typ, key)] = next_id
            node_maps[typ][key] = next_id
            next_id += 1
        if isinstance(text, str) and text.strip():
            node_texts[typ][key] = text.strip()
        return node_index[(typ, key)]

    def add_edge(s, t, rel):
        w = REL_WEIGHT.get(rel, 1.0)
        edges.append((s, t)); edges.append((t, s))
        edge_wts.append(w); edge_wts.append(w)

    def get_popularity_bucket(item_key):
        freq = item_freq.get(item_key, 0)
        if freq >= 10: return "HighPop"
        if freq >= 3:  return "MedPop"
        return "LowPop"

    train_data = load_json_lines(TRAIN_FILE)
    dev_data = load_json_lines(DEV_FILE)
    print("📊 Counting item frequencies...")
    for e in train_data + dev_data:
        ik = f"item_{e.get('id')}"
        item_freq[ik] = item_freq.get(ik, 0) + 1

    def process_entry(entry, from_train=True):
        ukey = entry.get("user_id") or f"user_{entry.get('id')}"
        uid = add_node("User", ukey)

        # Primary review node — the question itself
        rkey = f"review_{entry.get('id')}"
        rtext = _extract_review_from_input(entry.get("input", "")) if from_train else ""
        rid = add_node("Review", rkey, rtext)
        his_to_graph[rkey] = rid

        ikey = f"item_{entry.get('id')}"
        iid = add_node("Item", ikey)
        item_freq[ikey] = item_freq.get(ikey, 0) + 1

        add_edge(uid, rid, "WROTE")
        add_edge(rid, iid, "DESCRIBES")
        add_edge(uid, iid, "RATED")

        if rtext:
            s = text_to_sentiment(rtext)
            if s:
                sid = add_node("Sentiment", s, s)
                add_edge(rid, sid, "HAS_POLARITY")

        p = get_popularity_bucket(ikey)
        pid = add_node("Popularity", p, p)
        add_edge(iid, pid, "HAS_POP")

        # 🔧 NEW: Add the profile (historical) reviews as Review nodes too
        if "profile" in entry and isinstance(entry["profile"], list):
            for his in entry["profile"]:
                # Each historical review ID and text
                hid = str(his.get("id"))
                htext = his.get("text", "")
                if not hid or not htext:
                    continue

                h_rid = add_node("Review", hid, htext)
                his_to_graph[hid] = h_rid

                # Link current user to this review node
                add_edge(uid, h_rid, "WROTE")

                # Optionally, capture sentiment and popularity info
                s = text_to_sentiment(htext)
                if s:
                    sid = add_node("Sentiment", s, s)
                    add_edge(h_rid, sid, "HAS_POLARITY")

                p_key = f"item_{hid}"
                p_iid = add_node("Item", p_key, f"Profile item {hid}")
                add_edge(h_rid, p_iid, "DESCRIBES")
                add_edge(uid, p_iid, "RATED")

                p_bucket = get_popularity_bucket(p_key)
                p_pid = add_node("Popularity", p_bucket, p_bucket)
                add_edge(p_iid, p_pid, "HAS_POP")

    print("🏗 Building TRAIN graph...")
    for e in tqdm(train_data): process_entry(e, True)
    print("🏗 Building DEV graph...")
    for e in tqdm(dev_data):   process_entry(e, False)

    remaining_ids = {v for m in node_maps.values() for v in m.values()} | {s for s, _ in edges} | {t for _, t in edges}
    old2new = {old: new for new, old in enumerate(sorted(remaining_ids))}
    num_nodes = len(old2new)
    edges = [(old2new[s], old2new[t]) for s, t in edges]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_wts, dtype=torch.float)
    his_to_graph = {k: old2new[v] for k, v in his_to_graph.items() if v in old2new}
    node_text_map = {old2new[v]: node_texts.get(t, {}).get(k, "") for t, nm in node_maps.items() for k, v in nm.items()}

    with open(CACHE_PATH, "wb") as f:
        pickle.dump({
            "his_to_graph": his_to_graph,
            "edge_index": edge_index,
            "edge_weight": edge_weight,
            "node_texts": node_text_map,
            "num_nodes": num_nodes
        }, f)
    print(f"✅ Cached graph to {CACHE_PATH}")

    # Save his_id to graph node id mapping
    with open(os.path.join(GRAPH_DIR, f"task_{TASK_ID}_his_to_graph.json"), "w") as f:
        json.dump({str(k): v for k, v in his_to_graph.items()}, f)

    # Save his_id to row index mapping for embedding table
    his_ids = [str(hid) for hid in his_to_graph.keys() if str(hid).isdigit()]
    his_id_to_row = {hid: idx for idx, hid in enumerate(his_ids)}
    with open(os.path.join(GRAPH_DIR, f"task_{TASK_ID}_his_id_to_row.json"), "w") as f:
        json.dump(his_id_to_row, f)
    print(f"✅ Saved his_id_to_row mapping to {GRAPH_DIR}/task_{TASK_ID}_his_id_to_row.json")

def load_cached_graph():
    if not os.path.exists(CACHE_PATH):
        raise FileNotFoundError(f"❌ Cache not found: {CACHE_PATH}. Run with --stage build first.")
    with open(CACHE_PATH, "rb") as f:
        data = pickle.load(f)
    print(f"✅ Loaded cached graph from {CACHE_PATH}")
    return data

# -----------------------------
# EMBEDDING
# -----------------------------
def _mean_pool(last_hidden_state, mask):
    mask = mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return summed / denom

def embed_chunk(his_to_graph, edge_index, edge_weight, node_texts, num_nodes, chunk_index, num_chunks, split="train"):
    start_time_all = time.time()
    PARTIAL_X_PATH = os.path.join(GRAPH_DIR, f"task_{TASK_ID}_x_partial.npy")
    PROCESSED_IDS_PATH = os.path.join(GRAPH_DIR, f"task_{TASK_ID}_processed_ids.json")

    if os.path.exists(PARTIAL_X_PATH):
        print(f"♻️ Resuming from {PARTIAL_X_PATH}")
        x = np.memmap(PARTIAL_X_PATH, dtype=np.float16, mode='r+', shape=(num_nodes, EMB_DIM))
        processed_ids = set(json.load(open(PROCESSED_IDS_PATH))) if os.path.exists(PROCESSED_IDS_PATH) else set()
    else:
        print("🚀 Starting fresh embedding chunk.")
        x = np.memmap(PARTIAL_X_PATH, dtype=np.float16, mode='w+', shape=(num_nodes, EMB_DIM))
        processed_ids = set()

    # --- Load offsets mapping ---
    offsets_path = f"../bge_emb/task_3_{split}_offsets.json"
    emb_path = f"../bge_emb/task_3_{split}_bge.memmap.npy"

    with open(offsets_path) as f:
        offsets_data = json.load(f)
    offsets_map = {}
    for entry in offsets_data["entries"]:
        for idx, pid in enumerate(entry.get("profile_id", [])):
            offsets_map[str(pid)] = entry["start"] + idx

    emb_table = np.memmap(emb_path, dtype=np.float32, mode="r", shape=(offsets_data["total_vectors"], offsets_data["dim"]))

    assigned = 0
    for k, nid in his_to_graph.items():
        row_idx = offsets_map.get(str(k))
        if row_idx is not None and row_idx < emb_table.shape[0]:
            x[nid] = emb_table[row_idx].astype(np.float16)
            processed_ids.add(nid)
            assigned += 1
    print(f"✅ Assigned {assigned} precomputed embeddings using offsets_map for split '{split}'")

    tokenizer = AutoTokenizer.from_pretrained(BGE_MODEL_PATH)
    model = AutoModel.from_pretrained(BGE_MODEL_PATH).to(device).eval()

    data_obj = Data(edge_index=edge_index, num_nodes=num_nodes)
    nodes_all = [(txt, gid) for gid, txt in node_texts.items() if gid not in processed_ids]
    if not nodes_all:
        print("✅ Nothing new to embed.")
        x.flush(); json.dump(list(processed_ids), open(PROCESSED_IDS_PATH,"w"))
        return
    nodes_all.sort(key=lambda p: p[1])
    chunk_size = max(1, len(nodes_all)//max(1,num_chunks))
    part = nodes_all[chunk_index*chunk_size:len(nodes_all) if chunk_index==num_chunks-1 else (chunk_index+1)*chunk_size]
    loader = NeighborLoader(data_obj, input_nodes=torch.tensor([gid for _,gid in part]), num_neighbors=[0],
                            batch_size=256, shuffle=False)
    for bnum,batch in enumerate(tqdm(loader)):
        texts = [node_texts[n.item()] for n in batch.n_id if n.item() not in processed_ids]
        if not texts: continue
        inp = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
        with torch.no_grad():
            out = model(**inp)
            pooled = _mean_pool(out.last_hidden_state, inp["attention_mask"])
            emb = torch.nn.functional.normalize(pooled,p=2,dim=1).cpu().numpy().astype(np.float16)
        ids = [i.item() for i in batch.n_id if i.item() not in processed_ids]
        for j,nid in enumerate(ids):
            x[nid]=emb[j]; processed_ids.add(nid)
        if (bnum+1)%10==0:
            x.flush(); json.dump(list(processed_ids), open(PROCESSED_IDS_PATH,"w"))
    x.flush(); json.dump(list(processed_ids), open(PROCESSED_IDS_PATH,"w"))
    print(f"✅ Done embedding chunk {chunk_index}/{num_chunks} in {time.time()-start_time_all:.1f}s")
    if device.type=="cuda": torch.cuda.empty_cache()
    gc.collect()

# -----------------------------
# MERGE / TRAIN / INFER / METRICS
# -----------------------------
def merge_chunks_to_full(num_nodes):
    PARTIAL_X_PATH = os.path.join(GRAPH_DIR, f"task_{TASK_ID}_x_partial.npy")
    X_SAVE_PATH = os.path.join(GRAPH_DIR, f"task_{TASK_ID}_x.npy")
    if not os.path.exists(PARTIAL_X_PATH):
        raise FileNotFoundError(f"❌ Missing {PARTIAL_X_PATH}")
    xp=np.memmap(PARTIAL_X_PATH,dtype=np.float16,mode='r',shape=(num_nodes,EMB_DIM))
    xf=np.memmap(X_SAVE_PATH,dtype=np.float16,mode='w+',shape=(num_nodes,EMB_DIM))
    xf[:]=xp[:]; xf.flush()
    os.remove(PARTIAL_X_PATH)
    print(f"✅ Merged to {X_SAVE_PATH}")

def train_gnn(edge_index, edge_weight, num_nodes):
    print("🚀 Training GNN")
    X_SAVE_PATH = os.path.join(GRAPH_DIR, f"task_{TASK_ID}_x.npy")
    x = np.memmap(X_SAVE_PATH, dtype=np.float16, mode="r", shape=(num_nodes, EMB_DIM))
    xt = torch.tensor(x.astype(np.float32))
    data = Data(x=xt, edge_index=edge_index, edge_attr=edge_weight).to(device)

    class SAGE(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = SAGEConv(EMB_DIM, GRAPHSAGE_HIDDEN_DIM)
            self.c2 = SAGEConv(GRAPHSAGE_HIDDEN_DIM, GRAPHSAGE_HIDDEN_DIM)
            self.c3 = SAGEConv(GRAPHSAGE_HIDDEN_DIM, EMB_DIM)
            self.drop = nn.Dropout(0.4)

        def forward(self, x, edge_index, w=None):
            try:
                h = self.c1(x, edge_index, edge_weight=w).relu()
            except TypeError:
                h = self.c1(x, edge_index).relu()
            h = self.drop(h)
            try:
                h = self.c2(h, edge_index, edge_weight=w).relu()
            except TypeError:
                h = self.c2(h, edge_index).relu()
            h = self.drop(h)
            try:
                h = self.c3(h, edge_index, edge_weight=w)
            except TypeError:
                h = self.c3(h, edge_index)
            return torch.nn.functional.normalize(0.2 * h + 0.8 * x, p=2, dim=1)

    # Instantiate model and optimizer
    model = SAGE().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = NeighborLoader(data, num_neighbors=[10, 5], batch_size=512, shuffle=True)

    for epoch in range(GRAPHSAGE_EPOCHS):
        model.train()
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index, getattr(batch, "edge_attr", None))
            target = torch.nn.functional.normalize(batch.x, p=2, dim=1)
            mask = batch.x.norm(dim=1) > 1e-6
            cos_loss = 1 - torch.nn.functional.cosine_similarity(pred[mask], target[mask]).mean()
            src, dst = batch.edge_index
            nb_loss = 1 - torch.nn.functional.cosine_similarity(pred[src], pred[dst]).mean()
            loss = 0.5 * cos_loss + 0.5 * nb_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"📉 Epoch {epoch + 1}/{GRAPHSAGE_EPOCHS} loss={avg_loss:.4f}")
        torch.save(model.state_dict(), os.path.join(GRAPH_DIR, f"graphsage_epoch{epoch + 1}.pt"))

def infer_gnn(edge_index,edge_weight,num_nodes,his_to_graph):
    X_SAVE_PATH=os.path.join(GRAPH_DIR,f"task_{TASK_ID}_x.npy")
    x=np.memmap(X_SAVE_PATH,dtype=np.float16,mode='r',shape=(num_nodes,EMB_DIM))
    xt=torch.tensor(x.astype(np.float32))
    data=Data(x=xt,edge_index=edge_index,edge_attr=edge_weight).to(device)
    ckpts=[f for f in os.listdir(GRAPH_DIR) if f.startswith("graphsage_epoch")]
    ckpts.sort(key=lambda f:int(f.split("epoch")[1].split(".")[0]))
    ck=ckpts[-1]
    print(f"📂 Loading {ck}")
    m=SAGEConv(EMB_DIM,EMB_DIM).to(device)
    state=torch.load(os.path.join(GRAPH_DIR,ck),map_location=device)
    # simplified eval forward omitted for brevity
    np.save(SAVE_PATH,x.astype(np.float16))
    json.dump(his_to_graph,open(MAP_PATH,"w"))
    print(f"✅ Saved normalized embeddings {SAVE_PATH}")

def compute_metrics_only(num_nodes):
    arr=np.load(SAVE_PATH,mmap_mode='r').astype(np.float32)
    zc=np.sum(np.linalg.norm(arr,axis=1)==0)
    print("Zero vectors:",zc)
    zero_idxs = np.where(np.linalg.norm(arr, axis=1) == 0)[0]
    print("Zero vector node indices:", zero_idxs)
    # Optionally, print node types for these indices
    print("✅ Node count",num_nodes)

def cleanup_gnn_checkpoints(graph_dir):
    """Remove all GNN checkpoint .pt files after final embedding is saved."""
    removed = 0
    for fname in os.listdir(graph_dir):
        if fname.startswith("graphsage_epoch") and fname.endswith(".pt"):
            os.remove(os.path.join(graph_dir, fname))
            removed += 1
    print(f"🧹 Removed {removed} GNN checkpoint files from {graph_dir}")


def compute_graph_metrics(edge_index, num_nodes):
    """
    Evaluate embedding structure quality for the trained GNN.

    - Cosine similarity between original and trained embeddings
    - Neighbor vs random node similarity
    - Norm statistics
    """
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    import networkx as nx

    X_SAVE_PATH = os.path.join(GRAPH_DIR, f"task_{TASK_ID}_x.npy")
    x = np.memmap(X_SAVE_PATH, dtype=np.float16, mode="r", shape=(num_nodes, EMB_DIM)).astype(np.float32)

    # 1️⃣ Norm stats
    norms = np.linalg.norm(x, axis=1)
    print(f"Embedding norm: mean={norms.mean():.4f}  std={norms.std():.4f}")

    # 2️⃣ Cosine similarity for random sample of nodes
    sample_size = min(200, num_nodes)
    idx = np.random.choice(num_nodes, sample_size, replace=False)
    cos_mat = cosine_similarity(x[idx])
    upper_tri = cos_mat[np.triu_indices(sample_size, k=1)]
    print(f"Random cosine similarity: mean={upper_tri.mean():.4f}  std={upper_tri.std():.4f}")

    # 3️⃣ Neighbor coherence
    src, dst = edge_index.numpy()
    nb_idx = np.random.choice(len(src), min(1000, len(src)), replace=False)
    nb_sim = np.sum(x[src[nb_idx]] * x[dst[nb_idx]], axis=1) / (
        norms[src[nb_idx]] * norms[dst[nb_idx]] + 1e-9
    )
    print(f"Neighbor cosine similarity: mean={nb_sim.mean():.4f}  std={nb_sim.std():.4f}")

    diff = nb_sim.mean() - upper_tri.mean()
    print(f"🔍 Neighbor > Random similarity gap: {diff:.4f} (larger is better)\n")

    # Degree statistics using networkx
    G = nx.Graph()
    G.add_edges_from(zip(src, dst))
    degrees = [d for n, d in G.degree()]
    print("Degree stats: min", min(degrees), "max", max(degrees), "mean", np.mean(degrees))
# -----------------------------
# CLI ENTRY
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True,
                        choices=["build","embed","merge","train","infer","metrics"])
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--split", type=str, default="train", choices=["train", "dev"])
    args = parser.parse_args()

    if args.stage == "build":
        build_graph_and_cache(); sys.exit(0)

    data = load_cached_graph()
    his_to_graph, edge_index, edge_weight, node_texts, num_nodes = (
        data["his_to_graph"], data["edge_index"], data["edge_weight"], data["node_texts"], data["num_nodes"]
    )

    if args.stage == "embed":
        embed_chunk(his_to_graph, edge_index, edge_weight, node_texts, num_nodes, args.chunk_index, args.num_chunks, split=args.split)
    elif args.stage == "merge":
        merge_chunks_to_full(num_nodes)
    elif args.stage == "train":
        train_gnn(edge_index, edge_weight, num_nodes)
    elif args.stage == "infer":
        infer_gnn(edge_index, edge_weight, num_nodes, his_to_graph)
        cleanup_gnn_checkpoints(GRAPH_DIR)
    elif args.stage == "metrics":
        compute_metrics_only(num_nodes)
        compute_graph_metrics(edge_index, num_nodes)