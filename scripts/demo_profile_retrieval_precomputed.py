import json
import numpy as np

# --- CONFIG ---
JSON_PATH = "../LaMP_time_3_subset/train_questions.json"
EMB_PATH = "../bge_emb/task_3_train_bge.npy"
TOP_K = 5
ENTRY_IDX = 1  # Change this to select a different entry

# --- LOAD DATA ---
with open(JSON_PATH, "r") as f:
    data = json.load(f)

# --- LOAD EMBEDDINGS ---
embeddings = np.load(EMB_PATH)  # shape: (num_texts, emb_dim)

# --- Calculate offsets ---
offset = 0
for i in range(ENTRY_IDX):
    offset += len(data[i]["profile"]) + 1  # +1 for input

entry = data[ENTRY_IDX]
input_text = entry["input"]
profile_texts = [his["text"] for his in entry["profile"]]
num_profiles = len(profile_texts)

profile_embs = embeddings[offset : offset + num_profiles]
input_emb = embeddings[offset + num_profiles]

# --- SIMILARITY SEARCH ---
sims = np.dot(profile_embs, input_emb)
top_indices = np.argsort(-sims)[:TOP_K]

print(f"Input (question+review):\n{input_text}\n")
print("Profile texts:")
for i, txt in enumerate(profile_texts):
    print(f"  [{i}] {txt[:80]}...")

print("\nTop profile matches for input (using precomputed embeddings):")
for rank, idx in enumerate(top_indices):
    print(f"Rank {rank+1}: Score={sims[idx]:.3f}")
    print(f"  {profile_texts[idx]}\n")