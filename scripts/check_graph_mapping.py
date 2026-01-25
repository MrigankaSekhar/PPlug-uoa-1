import json
import numpy as np
import sys
import os

his_to_graph = json.load(open("../graph_emb/task_3_his_to_graph.json"))
graph_path = "../graph_emb/task_3_graph.npy"

# Get num_nodes from metadata
with open("../graph_emb/task_3_num_nodes.json") as f:
    num_nodes = int(json.load(f))
emb_dim = 768  # Make sure this matches your config

try:
    graph_emb = np.memmap(graph_path, dtype=np.float16, mode='r', shape=(num_nodes, emb_dim))
    print(f"Loaded graph embedding as raw memmap: shape={graph_emb.shape}, dtype={graph_emb.dtype}")
except Exception as e:
    print(f"❌ np.memmap failed: {e}")
    sys.exit(1)

max_node_id = graph_emb.shape[0] - 1

missing = []
for k, v in his_to_graph.items():
    try:
        node_id = int(v)
        if node_id < 0 or node_id > max_node_id:
            missing.append((k, v))
    except Exception as ex:
        print(f"Invalid node id for key {k}: {v} ({ex})")

print(f"Total missing graph node IDs: {len(missing)}")
if missing:
    print("Sample missing:", missing[:10])
else:
    print("All his_to_graph IDs are present in graph embedding memmap.")