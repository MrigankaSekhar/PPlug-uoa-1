import numpy as np
import json
import os

TASK_ID = 3
DATA_DIR = f"../LaMP_time_{TASK_ID}_subset_id"
DEV_PROFILE_FILE = os.path.join(DATA_DIR, "dev_profile.json")
HIS_EMB_PATH = f"../bge_emb/task_{TASK_ID}_bge.npy"
GRAPH_MAP_PATH = f"../graph_emb/task_{TASK_ID}_his_to_graph.json"
GRAPH_EMB_PATH = f"../graph_emb/task_{TASK_ID}_graph.npy"

def load_json_lines(path):
    with open(path) as f:
        try:
            return [json.loads(line) for line in f]
        except json.JSONDecodeError:
            f.seek(0)
            return json.load(f)

def check_all():
    print(f"🔍 Checking ID alignment for task {TASK_ID}…")
    miss_review_ids = []
    miss_graph_ids = []

    data = load_json_lines(DEV_PROFILE_FILE)
    his_embed = np.load(HIS_EMB_PATH, mmap_mode="r")
    graph_emb = np.load(GRAPH_EMB_PATH, mmap_mode="r")
    mapping = json.load(open(GRAPH_MAP_PATH))

    print(f"✅ Loaded {len(data)} examples from {DEV_PROFILE_FILE}")
    print(f"✅ his_embed shape: {his_embed.shape}, graph_embed shape: {graph_emb.shape}")
    print(f"✅ mapping entries: {len(mapping)}")

    def in_range(idx, max_len):
        return 0 <= idx < max_len

    for entry in data:
        for hid in entry.get("his_id", []):
            # his_id is now a synthetic integer index
            if not in_range(hid, his_embed.shape[0]):
                miss_review_ids.append(hid)
            mapped = mapping.get(str(hid))
            if mapped is None or not in_range(mapped, graph_emb.shape[0]):
                miss_graph_ids.append(hid)

    miss_review_ids = sorted(set(miss_review_ids))
    miss_graph_ids = sorted(set(miss_graph_ids))

    print(f"\n🚦 Missing in his_embed: {len(miss_review_ids)} / {sum(len(e.get('his_id', [])) for e in data)} profiles")
    if miss_review_ids:
        print("Example:", miss_review_ids[:10])
    print(f"🚦 Missing in graph_map or graph_emb: {len(miss_graph_ids)} / {sum(len(e.get('his_id', [])) for e in data)} profiles")
    if miss_graph_ids:
        print("Example:", miss_graph_ids[:10])

    if not miss_review_ids and not miss_graph_ids:
        print("\n✅ All IDs perfectly aligned between dev_profile, his embeddings, and graph embeddings!")
    else:
        print("\n⚠️ Detected ID misalignment — regenerate embeddings and graph using real review IDs.")

if __name__ == "__main__":
    check_all()