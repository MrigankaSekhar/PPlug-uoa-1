import os
import json
import time
import torch
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn # Import nn for the projection layer

# --------------------------
# CONFIG: dataset + schema
# --------------------------
TASK_ID = 3
GRAPH_DIR = "../graph_emb"
os.makedirs(GRAPH_DIR, exist_ok=True)
SAVE_PATH = os.path.join(GRAPH_DIR, f"task_{TASK_ID}_graph.emb")
MAP_PATH = os.path.join(GRAPH_DIR, f"task_{TASK_ID}_his_to_graph.json")

# Dataset files (change for other tasks)
TRAIN_FILE = f"../LaMP_time_{TASK_ID}_id/train_aug_input.json"
DEV_FILE = f"../LaMP_time_{TASK_ID}_id/dev_profile.json"

# Relation schema: (source_type, target_type, relation_name)
SCHEMA = [
    ("User", "Item", "RATED"),
    ("User", "Review", "WROTE"),
    ("Review", "Item", "DESCRIBES"),
    ("Item", "Category", "BELONGS_TO")
]

# Field map: tells loader where to pull each node key from dataset JSON
FIELD_MAP = {
    "User": "user_id",
    "Item": "item_id",
    "Review": "review_text",
    "Category": "category"
}

# Feature initialisation — choose: "random" or "bge"
FEATURE_INIT = "bge"  # or "bge/random"
BGE_MODEL_PATH = "../bge-base-en-v1.5"  # used if FEATURE_INIT="bge"

# Reduce EMB_DIM to save memory
EMB_DIM = 512  # significantly reduced from 768
BGE_OUTPUT_DIM = 768 # Original output dimension of the BGE model

# Determine device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --------------------------
# Load datasets
# --------------------------
def load_json_lines(path):
    with open(path) as f:
        try:
            return [json.loads(line) for line in f]
        except json.JSONDecodeError:
            # For pure JSON arrays
            f.seek(0)
            return json.load(f)

train_data = load_json_lines(TRAIN_FILE)
dev_data = load_json_lines(DEV_FILE)

# --------------------------
# Generic node & edge builders
# --------------------------
node_maps = {typ: {} for typ, _, _ in SCHEMA} | {tgt: {} for _, tgt, _ in SCHEMA}
counters = {typ: 1 for typ in node_maps}

his_to_graph = {}  # NEW: mapping from his_id to graph index

def add_node(typ, key):
    if key not in node_maps[typ]:
        node_maps[typ][key] = counters[typ]
        counters[typ] += 1
    return node_maps[typ][key]

edges = []

def process_entry(entry):
    """
    Process a single dataset entry and add edges based on SCHEMA and FIELD_MAP.
    This function is now domain-agnostic:
    - Uses FIELD_MAP to find the correct dataset keys for each node type.
    - Falls back to synthetic IDs if data is missing.
    """
    for src_type, tgt_type, rel_name in SCHEMA:
        # Get field names from FIELD_MAP or default to lowercase type name
        src_field = FIELD_MAP.get(src_type, src_type.lower())
        tgt_field = FIELD_MAP.get(tgt_type, tgt_type.lower()) # Fixed the UnboundLocalError here

        # Pull keys from dataset
        src_key = entry.get(src_field)
        tgt_key = entry.get(tgt_field)

        # Fallbacks for missing data: create synthetic node IDs
        if src_key is None:
            src_key = f"{src_type.lower()}_{hash(str(entry))}"
        if tgt_key is None:
            tgt_key = f"{tgt_type.lower()}_{hash(str(entry))}"

        # Register nodes and get unique integer IDs
        src_id = add_node(src_type, src_key)
        tgt_id = add_node(tgt_type, tgt_key)

        # Append edge in src→tgt
        edges.append((src_id, tgt_id))

# --------------------------
# Build graph from datasets
# --------------------------
for entry in train_data + dev_data:
    process_entry(entry)
    # Now also capture his_id mapping, if present
    if "his_id" in entry:
        for hid in entry["his_id"]:
            # Here we decide what graph node this history id should map to:
            # Example: if 'Review' nodes in SCHEMA correspond to history entries,
            # we link hid to the node index for that review
            graph_idx = add_node("Review", str(hid))
            his_to_graph[hid] = graph_idx

# Offset node IDs per type so they don't overlap
offsets = {}
current_offset = 0
for typ in node_maps:
    offsets[typ] = current_offset
    # Adjust mapping to global graph IDs
    for key, local_id in node_maps[typ].items():
        node_maps[typ][key] = local_id + current_offset
    current_offset += len(node_maps[typ]) + 1  # +1 for padding

edge_index_list = []
for src_id, tgt_id in edges:
    edge_index_list.append((src_id, tgt_id))
    edge_index_list.append((tgt_id, src_id))  # undirected

edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
num_nodes = current_offset

# --------------------------
# Paths for checkpointing
# --------------------------
X_SAVE_PATH = os.path.join(GRAPH_DIR, f"task_{TASK_ID}_x.pt")
PARTIAL_X_PATH = os.path.join(GRAPH_DIR, f"task_{TASK_ID}_x_partial.pt")
PROCESSED_IDS_PATH = os.path.join(GRAPH_DIR, f"task_{TASK_ID}_processed_ids.json")
MODEL_CKPT_PATH = os.path.join(GRAPH_DIR, "graphsage_epoch{}.pt")  # per epoch

# --------------------------
# Initialise node features with resume support
# --------------------------
if os.path.exists(X_SAVE_PATH):
    print(f"✅ Found saved node features: {X_SAVE_PATH}")
    x = torch.load(X_SAVE_PATH, map_location=device)
    # ✅ Ensure training uses float32 to avoid 'Half' CPU clamp error
    if x.dtype == torch.float16:
        x = x.to(torch.float32)

else:
    # If partial exists, load and resume from there
    processed_ids = set()
    if os.path.exists(PARTIAL_X_PATH):
        print(f"♻️ Resuming BGE embedding creation from partial checkpoint")
        x = torch.load(PARTIAL_X_PATH, map_location=device)
        if os.path.exists(PROCESSED_IDS_PATH):
            processed_ids = set(json.load(open(PROCESSED_IDS_PATH)))
    else:
        x = torch.zeros((num_nodes, EMB_DIM), device=device, dtype=torch.float16)
        processed_ids = set()

    if FEATURE_INIT == "random":
        x = torch.randn((num_nodes, EMB_DIM), device=device, dtype=torch.float16)

    elif FEATURE_INIT == "bge":
        tokenizer = AutoTokenizer.from_pretrained(BGE_MODEL_PATH)
        if device.type == 'cuda':
            model = AutoModel.from_pretrained(BGE_MODEL_PATH, torch_dtype=torch.float16).to(device).eval()
        else:
            model = AutoModel.from_pretrained(BGE_MODEL_PATH, torch_dtype=torch.float32).to(device).eval()

        projection_layer = None
        if EMB_DIM != BGE_OUTPUT_DIM:
            projection_layer = nn.Linear(BGE_OUTPUT_DIM, EMB_DIM).to(device)
            if device.type == 'cuda':
                projection_layer = projection_layer.half()

        text_nodes = [(key, idx) for typ in node_maps for key, idx in node_maps[typ].items() if isinstance(key, str)]
        batch_size = 256
        save_every = 100  # ✅ save partial checkpoint every 100 batches (~12,800 embeddings)

        # ✅ Ensure x uses float32 for training to avoid 'Half' clamp_min_scalar CPU error
        if x.dtype == torch.float16:
            x = x.to(torch.float32)

        for batch_num, i in enumerate(tqdm(range(0, len(text_nodes), batch_size), desc="Generating BGE embeddings")):
            batch_segment = text_nodes[i:i+batch_size]
            if all(idx in processed_ids for _, idx in batch_segment):
                continue  # skip already processed nodes

            batch_keys = [node[0] for node in batch_segment]
            batch_indices = [node[1] for node in batch_segment]
            inputs = tokenizer(batch_keys, return_tensors="pt", padding=True,
                               truncation=True, max_length=32).to(device)
            with torch.no_grad():
                embeddings = model(**inputs).last_hidden_state.mean(dim=1)
                if projection_layer:
                    embeddings = projection_layer(embeddings)

            # ✅ Keep embeddings in float32 during training
            for j, idx in enumerate(batch_indices):
                x[idx] = embeddings[j].to(torch.float32)
                processed_ids.add(idx)

            # ✅ Only save every `save_every` batches, not every time
            if (batch_num + 1) % save_every == 0 or (i + batch_size >= len(text_nodes)):
                torch.save(x, PARTIAL_X_PATH)
                json.dump(list(processed_ids), open(PROCESSED_IDS_PATH, "w"))
                print(f"💾 Partial checkpoint saved at batch {batch_num+1}")

        # Save final node features and clean partial files
        torch.save(x, X_SAVE_PATH)
        print(f"💾 Saved node features to {X_SAVE_PATH}")
        if os.path.exists(PARTIAL_X_PATH):
            os.remove(PARTIAL_X_PATH)
        if os.path.exists(PROCESSED_IDS_PATH):
            os.remove(PROCESSED_IDS_PATH)

        del tokenizer, model
        if projection_layer:
            del projection_layer
        torch.cuda.empty_cache()

data = Data(x=x, edge_index=edge_index).to(device)

# --------------------------
# GNN model
# --------------------------
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

gnn_model = GraphSAGE(in_channels=EMB_DIM, hidden_channels=128, out_channels=EMB_DIM).to(device)
if device.type == 'cuda':
    gnn_model = gnn_model.half()

optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.01)

# --------------------------
# ✅ Mini-batch training with NeighborLoader
# --------------------------
from torch_geometric.loader import NeighborLoader

loader = NeighborLoader(
    data,
    num_neighbors=[15, 10],   # sample neighbors at each hop
    batch_size=1024,          # adjust to your memory
    shuffle=True
)

# --------------------------
# Resume GNN training from last checkpoint
# --------------------------
start_epoch = 0
for file in os.listdir(GRAPH_DIR):
    if file.startswith("graphsage_epoch") and file.endswith(".pt"):
        ep_num = int(file[len("graphsage_epoch"):-3])
        if ep_num > start_epoch:
            start_epoch = ep_num

if start_epoch > 0:
    ckpt_path = MODEL_CKPT_PATH.format(start_epoch)
    print(f"🔄 Resuming training from {ckpt_path}")
    gnn_model.load_state_dict(torch.load(ckpt_path, map_location=device))

# --------------------------
# Training with per-epoch save
# --------------------------
for epoch in range(start_epoch, 10):
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
    print(f"💾 Saved checkpoint for epoch {epoch+1}")

# --------------------------
# Final save
# --------------------------
embeddings = gnn_model(data.x, data.edge_index).detach().cpu()
torch.save(embeddings, SAVE_PATH)
json.dump(his_to_graph, open(MAP_PATH, "w"))
print(f"✅ Saved graph embeddings to {SAVE_PATH} and mapping to {MAP_PATH}")
print(f"Shape: {embeddings.shape} — num_nodes={num_nodes}")