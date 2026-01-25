import json
import numpy as np

# --- CONFIG ---
JSON_PATH = "../LaMP_time_3_subset/train_questions.json"  # Path to the dataset
EMB_PATH = "../bge_emb/task_3_train_bge.memmap.npy"       # Path to the memmap embedding file
TOP_K = 5                                                 # Number of top matches to show
ENTRY_IDX = 212                                           # Index of the entry to analyze
emb_dim = 768                                             # Embedding dimension (must match how memmap was written)

# --- LOAD DATA ---
with open(JSON_PATH, "r") as f:
    data = json.load(f)  # List of dicts, each with "input", "profile", etc.

# --- Calculate total number of embedding vectors in memmap ---
# Each entry contributes len(profile) + 1 vectors: one for each profile item, plus one for the input/question.
num_texts = sum(len(entry["profile"]) + 1 for entry in data)

# --- LOAD EMBEDDINGS ---
# Read the memmap file as float32 (must match dtype used during writing).
# Shape: (num_texts, emb_dim)
embeddings = np.memmap(EMB_PATH, dtype=np.float32, mode="r", shape=(num_texts, emb_dim))

# --- Calculate offset for the selected entry ---
# The memmap is organized as: [profile_0, ..., profile_n, input] for each entry, in order.
# To find the start index for ENTRY_IDX, sum up all previous entries' (len(profile) + 1).
offset = 0
for i in range(ENTRY_IDX):
    offset += len(data[i]["profile"]) + 1  # +1 for input/question embedding

entry = data[ENTRY_IDX]
input_text = entry["input"]  # The question/review for this entry
profile_texts = [his["text"] for his in entry["profile"]]  # List of profile texts for this entry
num_profiles = len(profile_texts)

# --- Slice out the profile and input embeddings for the selected entry ---
# profile_embs: shape (num_profiles, emb_dim)
# input_emb: shape (emb_dim,)
profile_embs = embeddings[offset : offset + num_profiles]
input_emb = embeddings[offset + num_profiles]

# --- Defensive checks and normalization ---
def safe_normalize(vecs):
    """L2-normalize vectors along last axis, avoid division by zero."""
    norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
    return vecs / (norms + 1e-10)

# Check for NaN or zero vectors before similarity search
if np.isnan(profile_embs).any() or np.isnan(input_emb).any():
    print("❌ Embeddings contain NaN values! Check your memmap file and offset logic.")
    print("Profile embedding sample:", profile_embs[:2])
    print("Input embedding sample:", input_emb[:10])
elif np.linalg.norm(input_emb) == 0 or np.any(np.linalg.norm(profile_embs, axis=1) == 0):
    print("❌ Embeddings contain zero vectors! Check your memmap file and offset logic.")
    print("Profile embedding norm:", np.linalg.norm(profile_embs, axis=1))
    print("Input embedding norm:", np.linalg.norm(input_emb))
else:
    # Normalize embeddings to unit length (cosine similarity)
    profile_embs_norm = safe_normalize(profile_embs)
    input_emb_norm = safe_normalize(input_emb)
    # Compute cosine similarity between input and each profile embedding
    sims = np.dot(profile_embs_norm, input_emb_norm)
    # Get indices of top K most similar profiles
    top_indices = np.argsort(-sims)[:TOP_K]

    # --- Print results ---
    print(f"Input (question+review):\n{input_text}\n")
    print("Profile texts:")
    for i, txt in enumerate(profile_texts):
        print(f"  [{i}] {txt[:80]}...")  # Show first 80 chars for brevity

    print("\nTop profile matches for input (using precomputed embeddings):")
    for rank, idx in enumerate(top_indices):
        print(f"Rank {rank+1}: Score={sims[idx]:.3f}")
        print(f"  {profile_texts[idx]}\n")