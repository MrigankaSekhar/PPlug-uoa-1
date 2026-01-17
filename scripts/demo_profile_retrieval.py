import json
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

# --- CONFIG ---
MODEL_PATH = "../bge-base-en-v1.5"  # Change if needed
JSON_PATH = "../LaMP_time_3_subset/train_questions.json"
TOP_K = 5

# --- LOAD MODEL ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModel.from_pretrained(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()

def mean_pooling(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return summed / denom

def embed_texts(texts):
    with torch.no_grad():
        enc = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        out = model(**enc)
        pooled = mean_pooling(out.last_hidden_state, enc["attention_mask"])
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.cpu().numpy()

# --- LOAD DATA ---
with open(JSON_PATH, "r") as f:
    data = json.load(f)

# --- PICK A SAMPLE ENTRY ---
entry = data[1]  # Change index for other users
input_text = entry["input"]
profile_texts = [his["text"] for his in entry["profile"]]

print(f"Input (question+review):\n{input_text}\n")
print("Profile texts:")
for i, txt in enumerate(profile_texts):
    print(f"  [{i}] {txt[:80]}...")

# --- EMBED ---
all_texts = profile_texts + [input_text]
embeddings = embed_texts(all_texts)
profile_embs = embeddings[:-1]
input_emb = embeddings[-1]

# --- SIMILARITY SEARCH ---
sims = np.dot(profile_embs, input_emb)
top_indices = np.argsort(-sims)[:TOP_K]

print("\nTop profile matches for input:")
for rank, idx in enumerate(top_indices):
    print(f"Rank {rank+1}: Score={sims[idx]:.3f}")
    print(f"  {profile_texts[idx]}\n")