
"""
Optimized streaming embedding script for A100 GPU with configurable parameters.
This is a refactored version of embedding-slim-cuda.py where key parameters are externalised.
"""

import os
import json
import glob
import tempfile
import time
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# ==============================
# CONFIGURABLE PARAMETERS
# ==============================
TASK_IDS = [3]  # list of dataset task IDs to process
BASE_DIR = "."  # base directory where datasets live

MODEL_PATH = "./bge-base-en-v1.5"  # local or HF model repo
OUTPUT_DIR = "./bge_emb_config"    # folder to save embeddings
PRECISION = "fp16"                 # "fp16" or "bf16"
MAX_LENGTH = 512                   # max tokens per text
BATCH_SIZE = None                  # None = auto detect
DEVICE = "cpu:0"                    # "cuda" or "cpu"
CLEAN_TEMP = True                  # Delete temp batch files after merge

# ==============================
# DEVICE & MODEL INIT
# ==============================
# Detect device: prefer MPS if available, else CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
    default_batch = 64
    print("✅ Using Apple Silicon GPU via MPS with ..{default_batch} tokens per batch.")
elif torch.cuda.is_available() :
    device = torch.device(DEVICE)
    print(f"✅ Using CUDA device: {DEVICE}")
else:
    device = torch.device("cpu")
    default_batch = 256
    print("✅ Using CPU..{default_batch} tokens per batch.")

# assert torch.cuda.is_available(), "CUDA is required for this run."
# device = torch.device(DEVICE)

# Select dtype depending on device and precision
if device.type == "cuda":
    if PRECISION == "fp16":
        dtype = torch.float16
    elif PRECISION == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
else:
    # MPS or CPU fallback — layernorm & ops expect float32
    print(f"[precision] {device.type.upper()} does not fully support fp16 LayerNorm; using float32.")
    dtype = torch.float32

try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
except Exception:
    pass

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModel.from_pretrained(MODEL_PATH, torch_dtype=dtype).to(device).eval()

MODEL_DIM = getattr(model.config, "hidden_size", None) or 768

# ==============================
# HELPERS
# ==============================
def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return summed / denom

def auto_batch_size(model_dim, max_length, device):
    """
    Estimate batch size based on available GPU memory (CUDA).
    Falls back to a fixed safe batch size for CPU or MPS devices.
    """
    if device.type == "cuda":
        free_mem = torch.cuda.mem_get_info(device)[0] // (1024 ** 2)  # MB
        est_per_sample = max_length * model_dim * 2e-6
        safe_batch = int(free_mem / (est_per_sample * 1.5))
        return max(64, min(safe_batch, 2048))
    else:
        # Fallback for CPU or MPS — avoid CUDA-specific calls
        print(f"[auto_batch_size] Non-CUDA device detected: {device}. Using default=64")
        return default_batch

def move_batch_to_device(encodings, device):
    """
    Move batch tensors to the target device.
    Only use pin_memory for CUDA to enable async transfers.
    """
    if device.type == "cuda":
        return {k: v.pin_memory().to(device, non_blocking=True) for k, v in encodings.items()}
    else:
        return {k: v.to(device) for k, v in encodings.items()}

def process(idx, entry):
    question = entry["input"]
    profile = entry["profile"]
    new_entry = {}

    if (idx == 1):
        substr = 'title "'
        plc = question.find(substr) + len(substr)
        plc2 = question.find('",', plc)
        real_q = question[plc:plc2]
        new_entry["question"] = real_q
        new_entry["profile_txt"] = [his["title"] + " " + his["abstract"] for his in profile]
    elif (idx == 2):
        substr = "description: "
        plc = question.find(substr) + len(substr)
        real_q = question[plc:]
        new_entry["question"] = real_q
        new_entry["profile_txt"] = [his["description"] for his in profile]
    elif (idx == 3):
        substr = "review: "
        plc = question.find(substr) + len(substr)
        real_q = question[plc:]
        new_entry["question"] = real_q
        # Preserve both review text and explicit score with aligned profile IDs
        new_entry["profile_txt"] = []
        new_entry["profile_id"] = []
        for his in profile:
            new_entry["profile_txt"].append(
                f"Review: {his['text']} | Score: {his.get('score', '')}"
            )
            new_entry["profile_id"].append(his["id"])
    elif (idx == 4):
        substr = "article: "
        plc = question.find(substr) + len(substr)
        real_q = question[plc:]
        new_entry["question"] = real_q
        new_entry["profile_txt"] = [his["text"] + " " + his["title"] for his in profile]
    elif (idx == 5):
        substr = "paper: "
        plc = question.find(substr) + len(substr)
        real_q = question[plc:]
        new_entry["question"] = real_q
        new_entry["profile_txt"] = [his["title"] + " " + his["abstract"] for his in profile]
    elif (idx == 7):
        substr = "before or after it: "
        plc = question.find(substr) + len(substr)
        real_q = question[plc:]
        new_entry["question"] = real_q
        new_entry["profile_txt"] = [his["text"] for his in profile]
    new_entry["profile_id"] = [his["id"] for his in profile]
    return new_entry

def get_embedding_streaming(sentences, save_path, max_length=512, clean_temp=True):
    batch_size = BATCH_SIZE or auto_batch_size(MODEL_DIM, max_length, device)
    print(f"[embedding] Using batch_size={batch_size}")

    temp_dir = tempfile.mkdtemp(prefix="emb_batches_")
    existing_data = None
    start_item = 0

    if os.path.exists(save_path):
        try:
            existing_data = torch.load(save_path, map_location="cpu")
            if isinstance(existing_data, torch.Tensor):
                start_item = existing_data.shape[0]
                saved_dim = existing_data.shape[1] if existing_data.ndim >= 2 else None
                if saved_dim and saved_dim != MODEL_DIM:
                    raise ValueError(f"Dim mismatch: existing {saved_dim} vs model {MODEL_DIM}")
            print(f"[embedding] Resuming after {start_item} rows from {save_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load existing embeddings: {e}")
    else:
        print("[embedding] Starting fresh computation...")

    total_batches = len(sentences) // batch_size + (1 if len(sentences) % batch_size else 0)
    start_batch = start_item // batch_size
    start_time = time.time()

    for batch_idx in tqdm(range(start_batch, total_batches), initial=start_batch, total=total_batches, desc="Embedding batches"):
        s, e = batch_idx * batch_size, min((batch_idx + 1) * batch_size, len(sentences))

        enc = tokenizer(
            sentences[s:e],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        enc = move_batch_to_device(enc, device)
        ## normalization preserves semantic similarity in cosine space.
        with torch.inference_mode():
            out = model(**enc)
            pooled = mean_pool(out.last_hidden_state, enc["attention_mask"])
            pooled = F.normalize(pooled, p=2, dim=1)
            batch_emb = pooled.detach().to("cpu").float()

        torch.save(batch_emb, os.path.join(temp_dir, f"batch_{batch_idx}.pt"))
        del enc, out, pooled, batch_emb
        torch.cuda.empty_cache()

    batch_files = sorted(glob.glob(os.path.join(temp_dir, "*.pt")))
    final_parts = [torch.load(f, map_location="cpu") for f in batch_files]
    if existing_data is not None:
        final_parts.insert(0, existing_data)
    final_tensor = torch.cat(final_parts, dim=0).contiguous()
    torch.save(final_tensor, save_path)

    elapsed = time.time() - start_time
    print(f"✅ Saved {final_tensor.shape} embeddings to {save_path} in {elapsed:.2f}s")

    if clean_temp:
        for bf in batch_files: os.remove(bf)
        os.rmdir(temp_dir)

# ==============================
# MAIN
# ==============================
os.makedirs(OUTPUT_DIR, exist_ok=True)

for idx in TASK_IDS:
    dir_name = f"LaMP_time_{idx}"

    for split in ["train_questions.json", "dev_questions.json"]:
        file_path = os.path.join(BASE_DIR, dir_name, split)
        print(f"Processing {split} for task {idx}: {file_path}")
        dataset = json.load(open(file_path))
        all_txt_list = []
        for entry in dataset:
            ne = process(idx, entry)
            all_txt_list.extend(ne["profile_txt"] + [ne["question"]])

        save_path = os.path.join(OUTPUT_DIR, f"task_{idx}_{split.replace('.json','')}_bge.emb")
        get_embedding_streaming(all_txt_list, save_path, MAX_LENGTH, CLEAN_TEMP)
