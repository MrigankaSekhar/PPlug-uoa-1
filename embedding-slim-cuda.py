
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
import torch.backends.cuda
import torch.backends.cudnn

# -----------------------------
# DEVICE & PERFORMANCE SETTINGS
# -----------------------------
assert torch.cuda.is_available(), "CUDA is required for this run."
device = torch.device("cuda")

# Use FP16 for the model weights on A100; keep attention masks/int types untouched
dtype = torch.float16

# (Optional) A100 speedups on FP32 paths; harmless even if you mostly run FP16
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True  # good when input sizes are stable
except Exception:
    pass

# -----------------------------
# MODEL & TOKENIZER
# -----------------------------
# If you have the model locally, keep the local path; otherwise use "BAAI/bge-small-en-v1.5"
MODEL_PATH = "./bge-small-en-v1.5"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModel.from_pretrained(MODEL_PATH, torch_dtype=dtype).to(device).eval()

# Hidden size (embedding dimension) – safer than probing last_hidden_state directly
try:
    MODEL_DIM = getattr(model.config, "hidden_size", None)
except Exception:
    MODEL_DIM = None

# -----------------------------
# HELPERS
# -----------------------------
def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool over tokens using attention mask; returns [batch, hidden]"""
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # [B, T, 1]
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return summed / denom

def batch_to_device(batch, device):
    """Safely move BatchEncoding to device (works across HF versions)."""
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
def get_embedding_streaming(sentences, save_path, batch_size=128, max_length=512, clean_temp=True):
    """
    Fast & memory-safe streaming embed:
    - Tokenizes & runs embeddings in batches
    - Saves each batch to a temp file
    - Merges final tensor ONCE at the end
    - Resumes if save_path exists (shape must match model dim)
    """
    start_item = 0
    temp_dir = tempfile.mkdtemp(prefix="emb_batches_")
    print(f"[embedding] temp_dir: {temp_dir}")

    existing_data = None
    saved_dim = None

    # ---- Resume check
    if os.path.exists(save_path):
        try:
            existing_data = torch.load(save_path, map_location="cpu")
            if isinstance(existing_data, torch.Tensor):
                saved_dim = existing_data.shape[1] if existing_data.ndim >= 2 else existing_data.shape[0]
                start_item = existing_data.shape[0] if existing_data.ndim >= 2 else 0
            elif isinstance(existing_data, dict) and "embeddings" in existing_data and isinstance(existing_data["embeddings"], torch.Tensor):
                emb = existing_data["embeddings"]
                saved_dim = emb.shape[1]
                start_item = emb.shape[0]
            else:
                raise ValueError("Unsupported format in save_path: expected Tensor or dict['embeddings']")
        except Exception as e:
            raise RuntimeError(f"Failed to load existing embeddings from {save_path}: {e}")

        # Model dimension for validation
        current_dim = None
        if MODEL_DIM is not None:
            current_dim = MODEL_DIM
        else:
            # Guarded probe if config missing
            test_inputs = batch_to_device(tokenizer(["test"], return_tensors="pt"), device)
            with torch.inference_mode():
                test_outputs = model(**test_inputs)
            if hasattr(test_outputs, "last_hidden_state"):
                current_dim = test_outputs.last_hidden_state.shape[-1]
            elif isinstance(test_outputs, torch.Tensor):
                current_dim = test_outputs.shape[-1]
            else:
                raise RuntimeError("Cannot infer model output dimension.")

        if saved_dim is not None and saved_dim != current_dim:
            raise ValueError(f"Dim mismatch: existing file {saved_dim} vs model {current_dim}. Use a new save_path.")

        print(f"[embedding] Resuming after {start_item} rows from {save_path}")
    else:
        print("[embedding] Starting fresh computation...")

    total_batches = len(sentences) // batch_size + (1 if len(sentences) % batch_size else 0)
    start_batch = start_item // batch_size
    start_time = time.time()

    # ---- Process batches
    for batch_idx in tqdm(range(start_batch, total_batches), initial=start_batch, total=total_batches, desc="Embedding batches"):
        s = batch_idx * batch_size
        e = min(s + batch_size, len(sentences))

        enc = tokenizer(
            sentences[s:e],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        enc = batch_to_device(enc, device)

        with torch.inference_mode():
            out = model(**enc)  # BaseModelOutputWithPooling? (BGE uses a BERT-like backbone)
            last_hid = out.last_hidden_state  # [B, T, H]

            # Recommended BGE recipe: mean-pool over tokens -> L2 normalize
            pooled = mean_pool(last_hid, enc["attention_mask"])
            pooled = F.normalize(pooled, p=2, dim=1)  # [B, H], improve cosine similarity

            # Save CPU tensor (float32 for portability; change to half() if you prefer smaller files)
            batch_emb = pooled.detach().to("cpu").float()

        torch.save(batch_emb, os.path.join(temp_dir, f"batch_{batch_idx}.pt"))

        # Clear memory
        del enc, out, last_hid, pooled, batch_emb
        torch.cuda.empty_cache()

    # ---- Merge step once at end
    print("[embedding] Merging all batch files...")
    batch_files = sorted(glob.glob(os.path.join(temp_dir, "batch_*.pt")))
    all_emb = []
    if existing_data is not None:
        all_emb.append(existing_data if isinstance(existing_data, torch.Tensor) else existing_data["embeddings"])

    for bf in batch_files:
        bt = torch.load(bf, map_location="cpu")
        all_emb.append(bt)

    final_tensor = torch.cat(all_emb, dim=0).contiguous()
    torch.save(final_tensor, save_path)

    elapsed = time.time() - start_time
    print(f"✅ Completed: {final_tensor.shape[0]} embeddings saved to {save_path} in {elapsed:.2f}s ({elapsed/60:.2f} min)")

    # Optional cleanup
    if clean_temp:
        for bf in batch_files:
            try:
                os.remove(bf)
            except OSError:
                pass
        try:
            os.rmdir(temp_dir)
        except OSError:
            pass

# -----------------------------
# MAIN SCRIPT
# -----------------------------
os.makedirs("./bge_emb", exist_ok=True)  # folder to store all embeddings

# Only running LaMP-3 for now
for idx in [3]:
    dir_name = "LaMP_time_" + str(idx)

    # ---- TRAIN SET ----
    file_name = os.path.join(dir_name, "train_questions.json")
    print(f"Processing TRAIN set: {file_name}")
    dataset = json.load(open(file_name))
    all_txt_list = []
    for entry in dataset:
        ne = process(idx, entry)
        all_txt_list.extend(ne["profile_txt"] + [ne["question"]])

    get_embedding_streaming(
        all_txt_list,
        f"./bge_emb/task_{idx}_train_bge.emb",
        batch_size=512,   # tune if you see OOM; 512 is ok on 40GB with BGE-small
        max_length=512
    )

    # ---- DEV SET ----
    file_name = os.path.join(dir_name, "dev_questions.json")
    print(f"Processing DEV set: {file_name}")
    dataset = json.load(open(file_name))
    all_txt_list = []
    for entry in dataset:
        ne = process(idx, entry)
        all_txt_list.extend(ne["profile_txt"] + [ne["question"]])

    get_embedding_streaming(
        all_txt_list,
        f"./bge_emb/task_{idx}_dev_bge.emb",
        batch_size=512,
        max_length=512
    )
