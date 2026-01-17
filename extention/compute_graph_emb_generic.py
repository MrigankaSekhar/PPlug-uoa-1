import os
import json
import time
import argparse
import torch
import gc
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborLoader
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn

# --------------------------
# CONFIG
# --------------------------
TASK_ID = 3
GRAPH_DIR = "../graph_emb"
os.makedirs(GRAPH_DIR, exist_ok=True)
SAVE_PATH = os.path.join(GRAPH_DIR, f"task_{TASK_ID}_graph.emb")
MAP_PATH = os.path.join(GRAPH_DIR, f"task_{TASK_ID}_his_to_graph.json")

TRAIN_FILE = f"../LaMP_time_{TASK_ID}/train_questions.json"
DEV_FILE = f"../LaMP_time_{TASK_ID}/dev_questions.json"

SCHEMA = [
    ("User", "Item", "RATED"),
    ("User", "Review", "WROTE"),
    ("Review", "Item", "DESCRIBES"),
    ("Item", "Category", "BELONGS_TO")
]
FIELD_MAP = {
    "User": "user_id",
    "Item": "item_id",
    "Review": "review_text",
    "Category": "category"
}

FEATURE_INIT = "bge"
BGE_MODEL_PATH = "../bge-base-en-v1.5"
EMB_DIM = 768
BGE_OUTPUT_DIM = 768

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --------------------------
# UTILS
# --------------------------
def load_json_lines(path):
    with open(path) as f:
        try:
            return [json.loads(line) for line in f]
        except json.JSONDecodeError:
            f.seek(0)
            return json.load(f)

# --------------------------
# BUILD GRAPH STRUCTURE
# --------------------------
train_data = load_json_lines(TRAIN_FILE)
dev_data = load_json_lines(DEV_FILE)

node_maps = {typ: {} for typ, _, _ in SCHEMA} | {tgt: {} for _, tgt, _ in SCHEMA}
counters = {typ: 1 for typ in node_maps}
his_to_graph = {}
edges = []

def add_node(typ, key):
    if key not in node_maps[typ]:
        node_maps[typ][key] = counters[typ]
        counters[typ] += 1
    return node_maps[typ][key]

def process_entry(entry):
    for src_type, tgt_type, rel_name in SCHEMA:
        src_field = FIELD_MAP.get(src_type, src_type.lower())
        tgt_field = FIELD_MAP.get(tgt_type, tgt_type.lower())
        src_key = entry.get(src_field) or f"{src_type.lower()}_{hash(str(entry))}"
        tgt_key = entry.get(tgt_field) or f"{tgt_type.lower()}_{hash(str(entry))}"
        src_id = add_node(src_type, src_key)
        tgt_id = add_node(tgt_type, tgt_key)
        edges.append((src_id, tgt_id))

for entry in train_data + dev_data:
    process_entry(entry)
    if "his_id" in entry:
        for hid in entry["his_id"]:
            graph_idx = add_node("Review", str(hid))
            his_to_graph[hid] = graph_idx
    if "profile" in entry:
        for his in entry["profile"]:
            hid_val = his["id"]
            review_node = add_node("Review", str(hid_val))
            his_to_graph[hid_val] = review_node
            item_key = his.get("item_id") or f"item_{hash(his.get('text', '')[:50])}"
            item_node = add_node("Item", item_key)
            category_key = his.get("category")
            category_node = add_node("Category", category_key) if category_key else None
            user_key = entry.get("user_id") or f"user_{entry['id']}"
            user_node = add_node("User", user_key)
            edges.append((user_node, review_node))
            edges.append((review_node, item_node))
            if category_node is not None:
                edges.append((item_node, category_node))

# Offset node IDs
current_offset = 0
for typ in node_maps:
    for key, local_id in node_maps[typ].items():
        node_maps[typ][key] = local_id + current_offset
    current_offset += len(node_maps[typ]) + 1

edge_index_list = []
for src_id, tgt_id in edges:
    edge_index_list.append((src_id, tgt_id))
    edge_index_list.append((tgt_id, src_id))
edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
num_nodes = current_offset

X_SAVE_PATH = os.path.join(GRAPH_DIR, f"task_{TASK_ID}_x.pt")
PARTIAL_X_PATH = os.path.join(GRAPH_DIR, f"task_{TASK_ID}_x_partial.pt")
PROCESSED_IDS_PATH = os.path.join(GRAPH_DIR, f"task_{TASK_ID}_processed_ids.json")
MODEL_CKPT_PATH = os.path.join(GRAPH_DIR, "graphsage_epoch{}.pt")

# --------------------------
# STAGE FUNCTIONS
# --------------------------
def embed_chunk(chunk_index, num_chunks):
    # Load current partial state
    if os.path.exists(PARTIAL_X_PATH):
        print(f"♻️ Resuming from partial checkpoint: {PARTIAL_X_PATH}")
        x = torch.load(PARTIAL_X_PATH, map_location=device)
        processed_ids = set()
        if os.path.exists(PROCESSED_IDS_PATH):
            processed_ids = set(json.load(open(PROCESSED_IDS_PATH)))
            print(f"✅ Loaded {len(processed_ids)} processed node IDs from previous run")
    else:
        print("🚀 Starting fresh embedding for this chunk.")
        x = torch.zeros((num_nodes, EMB_DIM), device=device, dtype=torch.float32)
        processed_ids = set()

    # Precision selection
    if device.type == 'cuda':
        model_dtype = torch.float16
        print("⚡ Using GPU with FP16 precision for embeddings")
    else:
        model_dtype = torch.float32
        print("🖥️ Using CPU/MPS with FP32 precision for embeddings")

    tokenizer = AutoTokenizer.from_pretrained(BGE_MODEL_PATH)
    model = AutoModel.from_pretrained(
        BGE_MODEL_PATH,
        torch_dtype=model_dtype
    ).to(device).eval()

    # Bigger batches for GPU
    if device.type == 'cuda':
        batch_size = 4096   # adjust based on VRAM
        save_every = 200
    else:
        batch_size = 256
        save_every = 50
        
    projection_layer = None
    if EMB_DIM != BGE_OUTPUT_DIM:
        projection_layer = nn.Linear(BGE_OUTPUT_DIM, EMB_DIM).to(device)

    # Select only unprocessed nodes in this chunk
    text_nodes_all = [(key, idx) for typ in node_maps for key, idx in node_maps[typ].items() if isinstance(key, str)]
    chunk_size = len(text_nodes_all) // num_chunks
    start_i = chunk_index * chunk_size
    end_i = (chunk_index + 1) * chunk_size if chunk_index < num_chunks - 1 else len(text_nodes_all)
    text_nodes = text_nodes_all[start_i:end_i]
    text_nodes = [tn for tn in text_nodes if tn[1] not in processed_ids]

    print(f"📦 Processing chunk {chunk_index+1}/{num_chunks} — nodes {start_i} to {end_i-1}")
    print(f"⏩ Skipping {len(processed_ids)} previously processed nodes in this chunk")
    print(f"🎯 Remaining nodes to embed: {len(text_nodes)}")

    for batch_num, i in enumerate(tqdm(range(0, len(text_nodes), batch_size), desc="Embedding")):
        batch_segment = text_nodes[i:i+batch_size]
        batch_keys = [node[0] for node in batch_segment]
        batch_indices = [node[1] for node in batch_segment]

        inputs = tokenizer(batch_keys, return_tensors="pt", padding=True, truncation=True, max_length=32).to(device)
        with torch.no_grad():
            embeddings = model(**inputs).last_hidden_state.mean(dim=1)
            if projection_layer:
                embeddings = projection_layer(embeddings)

        for j, idx in enumerate(batch_indices):
            x[idx] = embeddings[j].to(torch.float32)
            processed_ids.add(idx)

        if (batch_num + 1) % save_every == 0 or (i + batch_size >= len(text_nodes)):
            torch.save(x, PARTIAL_X_PATH)
            json.dump(list(processed_ids), open(PROCESSED_IDS_PATH, "w"))
            print(f"💾 Partial checkpoint saved at batch {batch_num+1} — {len(processed_ids)} nodes done")

    # Final save of chunk progress
    torch.save(x, PARTIAL_X_PATH)
    json.dump(list(processed_ids), open(PROCESSED_IDS_PATH, "w"))
    print(f"✅ Chunk {chunk_index+1} completed — total {len(processed_ids)} nodes embedded in this chunk")

    del tokenizer, model
    if projection_layer:
        del projection_layer
    torch.cuda.empty_cache()
    gc.collect()

def merge_chunks_to_full():
    print("🔄 Merging partial embeddings into full X_SAVE_PATH...")
    if not os.path.exists(PARTIAL_X_PATH):
        raise FileNotFoundError(f"❌ Partial embeddings file not found: {PARTIAL_X_PATH}")
    x_partial = torch.load(PARTIAL_X_PATH, map_location="cpu")
    torch.save(x_partial.to(torch.float32), X_SAVE_PATH)
    print(f"✅ Merged and saved final embeddings to {X_SAVE_PATH}")

def train_gnn():
    start_time = time.time()
    x = torch.load(X_SAVE_PATH, map_location=device)
    data_obj = Data(x=x, edge_index=edge_index).to(device)

    class GraphSAGE(torch.nn.Module):
        def __init__(self, in_c, h_c, out_c):
            super().__init__()
            self.conv1 = SAGEConv(in_c, h_c)
            self.conv2 = SAGEConv(h_c, out_c)
        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index).relu()
            x = self.conv2(x, edge_index)
            return x

    gnn_model = GraphSAGE(EMB_DIM, 128, EMB_DIM).to(device)
    if device.type == 'cuda':
        gnn_model = gnn_model.half()
    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.01)

    loader = NeighborLoader(data_obj, num_neighbors=[15, 10], batch_size=256, shuffle=True)
    for epoch in range(4):
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            out = gnn_model(batch.x, batch.edge_index)
            loss = torch.nn.functional.mse_loss(out, batch.x.expand_as(out))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}: loss={total_loss/len(loader):.4f}")
        torch.save(gnn_model.state_dict(), MODEL_CKPT_PATH.format(epoch+1))

def infer_gnn():
    x = torch.load(X_SAVE_PATH, map_location=device)
    data_obj = Data(x=x, edge_index=edge_index).to(device)

    class GraphSAGE(torch.nn.Module):
        def __init__(self, in_c, h_c, out_c):
            super().__init__()
            self.conv1 = SAGEConv(in_c, h_c)
            self.conv2 = SAGEConv(h_c, out_c)
        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index).relu()
            x = self.conv2(x, edge_index)
            return x

    ckpts = sorted(
        [f for f in os.listdir(GRAPH_DIR) if f.startswith("graphsage_epoch") and f.endswith(".pt")],
        key=lambda f: int(f.split("epoch")[1].split(".")[0])
    )
    if not ckpts:
        raise FileNotFoundError("❌ No GNN checkpoints found — run the train stage first.")
    latest_ckpt = ckpts[-1]
    print(f"📂 Loading latest checkpoint: {latest_ckpt}")

    gnn_model = GraphSAGE(EMB_DIM, 128, EMB_DIM).to(device)
    gnn_model.load_state_dict(torch.load(os.path.join(GRAPH_DIR, latest_ckpt), map_location=device))
    gnn_model.eval()

    loader = NeighborLoader(data_obj, num_neighbors=[-1], batch_size=256, shuffle=False)
    final_embeddings = torch.zeros((num_nodes, EMB_DIM), dtype=torch.float32)

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Final inference"):
            out = gnn_model(batch.x, batch.edge_index).cpu()
            final_embeddings[batch.n_id] = out

    torch.save(final_embeddings, SAVE_PATH)
    json.dump(his_to_graph, open(MAP_PATH, "w"))
    print(f"✅ Saved graph embeddings to {SAVE_PATH}")

# --------------------------
# CLI ENTRY
# --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["embed", "merge", "train", "infer"])
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

# ... inside your final inference stage after you have X (tensor of embeddings)

import numpy as np

if __name__ == "__main__":
    # ... after infer_gnn or final embedding computation
    final_array = x.cpu().numpy().astype(np.float16)  # Half precision
    np.save(SAVE_PATH.replace(".emb", ".npy"), final_array)
    print(f"✅ Saved FP16 memory-mapped embeddings to {SAVE_PATH.replace('.emb', '.npy')}")
    np.save(SAVE_PATH, final_array.astype(np.float16))
    print(f"✅ Saved graph embeddings to {SAVE_PATH} as NumPy .npy format")