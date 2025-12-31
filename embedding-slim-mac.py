from rank_bm25 import BM25Okapi
import os
import json
import re
from tqdm import tqdm
import time
import tempfile
import glob

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# ✅ Subset toggle
USE_SUBSET = True  # change to False to use full dataset

# -----------------------------
# DEVICE & PERFORMANCE SETTINGS
# -----------------------------
# Detect device: prefer MPS if available, else CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ Using Apple Silicon GPU via MPS")
else:
    device = torch.device("cpu")
    print("✅ Using CPU")

# Use float32 for CPU/MPS
dtype = torch.float32

# -----------------------------
# MODEL & TOKENIZER
# -----------------------------
MODEL_PATH = "./bge-base-en-v1.5"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModel.from_pretrained(MODEL_PATH, torch_dtype=dtype).to(device).eval()

try:
    MODEL_DIM = getattr(model.config, "hidden_size", None)
except Exception:
    MODEL_DIM = None

# -----------------------------
# HELPERS
# -----------------------------
def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return summed / denom

def batch_to_device(batch, device):
    if hasattr(batch, "to"):
        try:
            return batch.to(device)
        except Exception:
            pass
    return {k: v.to(device) for k, v in batch.items()}

# -----------------------------
# DATA PROCESSING FUNCTION
# -----------------------------
def process(idx, entry):
    question = entry["input"]
    profile = entry["profile"]
    new_entry = {}

    if (idx == 1):
        substr = 'title "'
        plc = question.find(substr) + len(substr)
        substr2 = '",'
        plc2 = question.find(substr2, plc)
        real_question = question[plc:plc2]
        new_entry["question"] = real_question
        new_entry["profile_txt"] = [his["title"] + " " + his["abstract"] for his in profile]
        new_entry["profile_id"] = [his["id"] for his in profile]

    if (idx == 2):
        substr = "description: "
        plc = question.find(substr) + len(substr)
        real_question = question[plc:]
        new_entry["question"] = real_question
        new_entry["profile_txt"] = [his["description"] for his in profile]
        new_entry["profile_id"] = [his["id"] for his in profile]

    if (idx == 3):
        substr = "review: "
        plc = question.find(substr) + len(substr)
        real_question = question[plc:]
        new_entry["question"] = real_question
        new_entry["profile_txt"] = [his["text"] for his in profile]
        new_entry["profile_id"] = [his["id"] for his in profile]

    if (idx == 4):
        substr = "article: "
        plc = question.find(substr) + len(substr)
        real_question = question[plc:]
        new_entry["question"] = real_question
        new_entry["profile_txt"] = [his["text"] + " " + his["title"] for his in profile]
        new_entry["profile_id"] = [his["id"] for his in profile]

    if (idx == 5):
        substr = "paper: "
        plc = question.find(substr) + len(substr)
        real_question = question[plc:]
        new_entry["question"] = real_question
        new_entry["profile_txt"] = [his["title"] + " " + his["abstract"] for his in profile]
        new_entry["profile_id"] = [his["id"] for his in profile]

    if (idx == 7):
        substr = "before or after it: "
        plc = question.find(substr) + len(substr)
        real_question = question[plc:]
        new_entry["question"] = real_question
        new_entry["profile_txt"] = [his["text"] for his in profile]
        new_entry["profile_id"] = [his["id"] for his in profile]

    return new_entry

# -----------------------------
# STREAMING EMBEDDING FUNCTION
# -----------------------------
def get_embedding_streaming(sentences, save_path, batch_size=64, max_length=512, clean_temp=True):
    start_item = 0
    temp_dir = tempfile.mkdtemp(prefix="emb_batches_")
    print(f"[embedding] temp_dir: {temp_dir}")

    existing_data = None
    saved_dim = None

    if os.path.exists(save_path):
        try:
            existing_data = torch.load(save_path, map_location="cpu")
            if isinstance(existing_data, torch.Tensor):
                saved_dim = existing_data.shape[1] if existing_data.ndim >= 2 else existing_data.shape[0]
                start_item = existing_data.shape[0] if existing_data.ndim >= 2 else 0
        except Exception as e:
            raise RuntimeError(f"Failed to load existing embeddings from {save_path}: {e}")

        current_dim = MODEL_DIM
        if saved_dim is not None and saved_dim != current_dim:
            raise ValueError(f"Dim mismatch: existing file {saved_dim} vs model {current_dim}")
        print(f"[embedding] Resuming after {start_item} rows from {save_path}")
    else:
        print("[embedding] Starting fresh computation...")

    total_batches = len(sentences) // batch_size + (1 if len(sentences) % batch_size else 0)
    start_batch = start_item // batch_size
    start_time = time.time()

    for batch_idx in tqdm(range(start_batch, total_batches), initial=start_batch, total=total_batches):
        s, e = batch_idx * batch_size, min((batch_idx + batch_size), len(sentences))
        enc = tokenizer(sentences[s:e], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        enc = batch_to_device(enc, device)

        with torch.inference_mode():
            out = model(**enc)
            pooled = mean_pool(out.last_hidden_state, enc["attention_mask"])
            pooled = F.normalize(pooled, p=2, dim=1)
            batch_emb = pooled.detach().to("cpu").float()

        torch.save(batch_emb, os.path.join(temp_dir, f"batch_{batch_idx}.pt"))
        del enc, out, pooled, batch_emb

    print("[embedding] Merging all batch files...")
    batch_files = sorted(glob.glob(os.path.join(temp_dir, "batch_*.pt")))
    all_emb = []
    if existing_data is not None:
        all_emb.append(existing_data)
    for bf in batch_files:
        all_emb.append(torch.load(bf, map_location="cpu"))
    final_tensor = torch.cat(all_emb, dim=0).contiguous()
    torch.save(final_tensor, save_path)

    elapsed = time.time() - start_time
    print(f"✅ Completed: {final_tensor.shape[0]} embeddings saved to {save_path} in {elapsed:.2f}s")

    if clean_temp:
        for bf in batch_files:
            os.remove(bf)
        os.rmdir(temp_dir)

# -----------------------------
# MAIN SCRIPT
# -----------------------------
os.makedirs("./bge_emb", exist_ok=True)

for idx in [3]:
    if USE_SUBSET:
        dir_name = f"LaMP_time_{idx}_subset"
    else:
        dir_name = f"LaMP_time_{idx}"

    print(f"📄 Dataset directory: {dir_name}")

    print(f"Processing TRAIN set: {dir_name}")
    dataset = json.load(open(os.path.join(dir_name, "train_questions.json")))
    all_txt_list = []
    for entry in dataset:
        ne = process(idx, entry)
        all_txt_list.extend(ne["profile_txt"] + [ne["question"]])
    get_embedding_streaming(all_txt_list, f"./bge_emb/task_{idx}_train_bge.emb")

    print(f"Processing DEV set: {dir_name}")
    dataset = json.load(open(os.path.join(dir_name, "dev_questions.json")))
    all_txt_list = []
    for entry in dataset:
        ne = process(idx, entry)
        all_txt_list.extend(ne["profile_txt"] + [ne["question"]])
    get_embedding_streaming(all_txt_list, f"./bge_emb/task_{idx}_dev_bge.emb")