
import os
import json
import time
import torch
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# --------------------------
# CONFIG: dataset + schema
# --------------------------
TASK_ID = 3
GRAPH_DIR = "../graph_emb"
os.makedirs(GRAPH_DIR, exist_ok=True)
SAVE_PATH = os.path.join(GRAPH_DIR, f"task_{TASK_ID}_graph.emb")

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
FEATURE_INIT = "random"  # or "bge/random"
BGE_MODEL_PATH = "../bge-base-en-v1.5"  # used if FEATURE_INIT="bge"

EMB_DIM = 768  # match the embedding size used in your model

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
        tgt_field = FIELD_MAP.get(tgt_type, tgt_type.lower())

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

# Offset node IDs per type so they don't overlap
offsets = {}
current_offset = 0
for typ in node_maps:
    offsets[typ] = current_offset
    current_offset += len(node_maps[typ]) + 1  # +1 for padding

edge_index_list = []
for src_id, tgt_id in edges:
    edge_index_list.append((src_id, tgt_id))
    edge_index_list.append((tgt_id, src_id))  # undirected

edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
num_nodes = current_offset

# --------------------------
# Initialise node features
# --------------------------
if FEATURE_INIT == "random":
    x = torch.randn((num_nodes, EMB_DIM))
elif FEATURE_INIT == "bge":
    tokenizer = AutoTokenizer.from_pretrained(BGE_MODEL_PATH)
    model = AutoModel.from_pretrained(BGE_MODEL_PATH, torch_dtype=torch.float32).eval()
    x = torch.zeros((num_nodes, EMB_DIM))
    for typ in node_maps:
        for key, idx in node_maps[typ].items():
            inputs = tokenizer(key, return_tensors="pt", max_length=32, truncation=True)
            with torch.no_grad():
                outputs = model(**inputs)
                emb = outputs.last_hidden_state.mean(dim=1).squeeze(0)
            x[idx] = emb

data = Data(x=x, edge_index=edge_index)

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

gnn_model = GraphSAGE(in_channels=EMB_DIM, hidden_channels=128, out_channels=EMB_DIM)
optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.01)

gnn_model.train()
start_time = time.time()
epoch_times = []
epochs = 100

for epoch in range(epochs):
    epoch_start = time.time()

    optimizer.zero_grad()
    out = gnn_model(data.x, data.edge_index)
    loss = torch.nn.functional.mse_loss(out, data.x.expand_as(out))
    loss.backward()
    optimizer.step()

    # Store elapsed time for this epoch
    epoch_duration = time.time() - epoch_start
    epoch_times.append(epoch_duration)

    # Compute ETA based on average epoch_duration so far
    avg_time = sum(epoch_times) / len(epoch_times)
    remaining_epochs = epochs - (epoch + 1)
    eta_seconds = avg_time * remaining_epochs

    if epoch % 10 == 0:
        # Format ETA nicely as mm:ss
        eta_formatted = time.strftime("%M:%S", time.gmtime(eta_seconds))
        print(f"Epoch {epoch}: loss={loss.item():.4f} | ETA: {eta_formatted}")

total_duration = time.time() - start_time
print(f"⏱ Total training time: {time.strftime('%M:%S', time.gmtime(total_duration))}")

# --------------------------
# Save embeddings
# --------------------------
embeddings = gnn_model(data.x, data.edge_index).detach()
torch.save(embeddings, SAVE_PATH)
print(f"✅ Saved graph embeddings to {SAVE_PATH}")
print(f"Shape: {embeddings.shape} — num_nodes={num_nodes}")
