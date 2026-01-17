import json, os, numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import pickle 

# ============================================================
# 🔧 CONFIG — aligned with new graph pipeline setup
# ============================================================
TASK_ID = 3
GRAPH_DIR = "../graph_emb"
DATA_DIR = f"../LaMP_time_{TASK_ID}_subset"

JSON_PATH = os.path.join(DATA_DIR, "train_questions.json")
GRAPH_PATH = os.path.join(GRAPH_DIR, f"task_{TASK_ID}_x.npy")          # new embedding file from pipeline
HIS_TO_GRAPH_PATH = os.path.join(GRAPH_DIR, f"task_{TASK_ID}_his_to_graph.json")
CACHE_PATH = os.path.join(GRAPH_DIR, f"task_{TASK_ID}_graph_cache.pkl")  # node text/id info
BGE_MODEL_PATH = "../bge-base-en-v1.5"
TOP_K = 5
ENTRY_IDX = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# LOAD GRAPH + DATA
# ============================================================
with open(JSON_PATH, "r") as f:
    data = json.load(f)

with open(CACHE_PATH, "rb") as f:
    graph_data = pickle.load(f)
num_nodes = graph_data.get("num_nodes")
emb_graph = np.memmap(GRAPH_PATH, dtype=np.float16, mode="r", shape=(num_nodes, 768))
print(f"✅ Loaded memmap graph with shape {emb_graph.shape}")

with open(HIS_TO_GRAPH_PATH, "r") as f:
    his_to_graph = {str(k): int(v) for k, v in json.load(f).items()}

# Try to load cached graph info (node text map + num_nodes)
if os.path.exists(CACHE_PATH):
    import pickle
    with open(CACHE_PATH, "rb") as f:
        graph_data = pickle.load(f)
    node_texts = graph_data.get("node_texts", {})
    num_nodes = graph_data.get("num_nodes", len(emb_graph))
    print(f"✅ Loaded cached graph info: {num_nodes} nodes")
else:
    node_texts = {}
    print("⚠️ No cached graph data found — continuing with embeddings only")

# ============================================================
# DEFINE INPUT: custom vs dataset
# ============================================================
CUSTOM_INPUT = """What is the score of the following review on a scale of 1 to 5? just answer with 
1, 2, 3, 4, or 5 without further explanation. review: It was easy to set up and worked right away. 
The build quality is excellent, and customer support was extremely helpful. Highly recommended!"""
USE_CUSTOM = True  # set False to fall back to ENTRY_IDX from JSON file

if USE_CUSTOM:
    entry = data[ENTRY_IDX]  # still use dataset item for profile
    input_text = CUSTOM_INPUT.strip()
else:
    entry = data[ENTRY_IDX]
    input_text = entry["input"].strip()

profile_texts = [his["text"] for his in entry.get("profile", [])]
profile_ids = [str(his["id"]) for his in entry.get("profile", [])]

# ============================================================
# COLLECT PROFILE EMBEDDINGS
# ============================================================
profile_embs_graph = []
for hid in profile_ids:
    node_id = his_to_graph.get(hid)
    if node_id is not None and node_id < emb_graph.shape[0]:
        profile_embs_graph.append(emb_graph[node_id])
    else:
        profile_embs_graph.append(np.zeros(emb_graph.shape[1], dtype=emb_graph.dtype))
profile_embs_graph = np.stack(profile_embs_graph) if profile_embs_graph else np.zeros((1, emb_graph.shape[1]))

# ============================================================
# EMBED CUSTOM INPUT TEXT (BGE)
# ============================================================
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


if USE_CUSTOM:
    print("\n🧠 Embedding custom input text with BGE...")
    input_emb_graph = embed_bge_text(input_text, BGE_MODEL_PATH, device=DEVICE)
else:
    query_id_str = str(entry["id"])
    node_id = his_to_graph.get(query_id_str)
    if node_id is not None:
        input_emb_graph = emb_graph[node_id]
    else:
        input_emb_graph = np.zeros(emb_graph.shape[1], dtype=emb_graph.dtype)

# ============================================================
# DIAGNOSTIC & PROFILE RETRIEVAL
# ============================================================
print("\nProfile graph embedding norms:")
for i, vec in enumerate(profile_embs_graph):
    print(f"Profile {i}: norm={np.linalg.norm(vec):.3f}")
print(f"Input graph embedding norm: {np.linalg.norm(input_emb_graph):.3f}")

if np.linalg.norm(input_emb_graph) == 0.0:
    print("\n⚠️ Zero embedding for input; similarity scores will all be zero.")
else:
    sims_graph = np.dot(profile_embs_graph, input_emb_graph)
    top_indices_graph = np.argsort(-sims_graph)[:TOP_K]

    print(f"\nInput text:\n{input_text}\n")
    print("Profile texts:")
    for i, txt in enumerate(profile_texts):
        print(f"  [{i}] {txt[:80]}...")

    print("\nTop profile matches for input (GNN-based Graph embeddings):")
    for rank, idx in enumerate(top_indices_graph):
        print(f"Rank {rank+1}: Score={sims_graph[idx]:.3f}")
        print(f"  {profile_texts[idx]}\n")