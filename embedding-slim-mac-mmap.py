import argparse, os, torch, numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import ijson
import json

USE_SUBSET = True  # change to False to use full dataset
idx = 3  # LaMP_time_3

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="./bge-base-en-v1.5")
parser.add_argument("--base_dir", type=str, default=".")
parser.add_argument("--chunk-index", type=int, default=0)
parser.add_argument("--num-chunks", type=int, default=1)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--max-length", type=int, default=128)
parser.add_argument("--merge-only", action="store_true", help="Only merge chunk files into final .bge.npy")
args = parser.parse_args()

# -----------------------------------------
# Merge step for per-split chunks
# -----------------------------------------
def merge_chunks():
    print("🔄 Merging chunk files into final bge npy...")
    for split in ["train", "dev"]:
        merged = []
        chunk_files = []
        for chunk_idx in range(args.num_chunks):
            chunk_path = f"./bge_emb/task_{idx}_{split}_chunk{chunk_idx}.npy"
            if not os.path.exists(chunk_path):
                raise FileNotFoundError(f"Missing chunk file: {chunk_path}")
            merged.append(np.load(chunk_path))
            chunk_files.append(chunk_path)
        final_arr = np.vstack(merged)
        final_path = f"./bge_emb/task_{idx}_{split}_bge.npy"
        np.save(final_path, final_arr)
        print(f"✅ Final merged file saved: {final_path} ({final_arr.shape[0]} rows)")
        # Clean up chunk files after merge
        for chunk_path in chunk_files:
            try:
                os.remove(chunk_path)
                print(f"🗑️ Deleted chunk file: {chunk_path}")
            except Exception as e:
                print(f"⚠️ Could not delete chunk file {chunk_path}: {e}")
    print("🎯 Merge complete.")

if args.merge_only:
    merge_chunks()
    exit(0)

# -----------------------------------------
# Device setup
# -----------------------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
dtype = torch.float32 if device.type == "mps" else torch.float16
print(f"✅ Using {device}, dtype={dtype}")

# Load tokenizer/model
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
model = AutoModel.from_pretrained(args.model_path, torch_dtype=dtype).to(device).eval()
MODEL_DIM = getattr(model.config, "hidden_size", 768)

# Disable torch.compile for MPS
try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    if hasattr(torch, "compile") and device.type == "cuda":
        model = torch.compile(model)
        print("⚡ Model compiled for CUDA speed")
    else:
        print("ℹ️ Skipping torch.compile (not supported for MPS/this torch version)")
except Exception:
    print("ℹ️ torch.compile failed, running eagerly")

# -----------------------------------------
# Helper functions
# -----------------------------------------
def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return summed / denom

def stream_dataset(json_path):
    with open(json_path, 'rb') as f:
        for entry in ijson.items(f, 'item'):
            yield entry

def entry_texts(entry):
    return [his["text"] for his in entry.get("profile", [])] + [entry.get("input", "")]

# -----------------------------------------
# Dataset root folder
# -----------------------------------------
if USE_SUBSET:
    dir_name = f"LaMP_time_{idx}_subset"
else:
    dir_name = f"LaMP_time_{idx}"
print(f"[embedding] Using dataset path: {dir_name}")

os.makedirs("./bge_emb", exist_ok=True)

# -----------------------------------------
# Embedding generation loop
# -----------------------------------------
for split in ["train", "dev"]:
    json_path = os.path.join(dir_name, f"{split}_questions.json")

    all_texts = []
    all_his_ids = []

    for entry in stream_dataset(json_path):
        for his in entry.get("profile", []):
            # Use REAL review IDs if available, fall back to synthetic ones
            real_id = str(his.get("id")) if "id" in his else str(len(all_his_ids))
            all_texts.append(his["text"])
            all_his_ids.append(real_id)

        # Optionally add the main input text as well
        all_texts.append(entry.get("input", ""))
        all_his_ids.append(None)  # we don't embed the query itself

    # Save mapping for profile reviews only
    his_id_to_row = {hid: idx for idx, hid in enumerate(all_his_ids) if hid is not None}
    map_path = f"./bge_emb/task_{idx}_{split}_bge_idmap.json"
    with open(map_path, "w") as f:
        json.dump(his_id_to_row, f)
    print(f"✅ Saved his_id to row mapping: {map_path}")

    total_rows = len(all_texts)
    chunk_size = (total_rows + args.num_chunks - 1) // args.num_chunks
    start_row = args.chunk_index * chunk_size
    end_row = min(start_row + chunk_size, total_rows)
    texts_chunk = all_texts[start_row:end_row]

    print(f"⚡ Chunk {args.chunk_index+1}/{args.num_chunks}: rows {start_row}-{end_row-1}")

    embeddings = np.zeros((len(texts_chunk), MODEL_DIM), dtype=np.float16)

    for i in tqdm(range(0, len(texts_chunk), args.batch_size), desc=f"Embedding({split})", unit="batch"):
        batch_texts = texts_chunk[i:i+args.batch_size]
        safe_max_len = min(args.max_length, tokenizer.model_max_length)
        enc = tokenizer(batch_texts, padding=True, truncation=True,
                        max_length=safe_max_len, return_tensors="pt")

        for k, v in enc.items():
            if k in ("input_ids", "attention_mask", "token_type_ids"):
                enc[k] = v.to(device)
            else:
                enc[k] = v.to(device, dtype=dtype)

        with torch.inference_mode():
            out = model(**enc)
            pooled = mean_pool(out.last_hidden_state, enc["attention_mask"].to(dtype))
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            emb_batch = pooled.cpu().numpy().astype(np.float16)

        embeddings[i:i+len(emb_batch)] = emb_batch
        del enc, out, pooled, emb_batch
        if device.type == "cuda":
            torch.cuda.empty_cache()

    chunk_file = f"./bge_emb/task_{idx}_{split}_chunk{args.chunk_index}.npy"
    np.save(chunk_file, embeddings)
    print(f"✅ Saved {len(texts_chunk)} rows to {chunk_file}")

# -----------------------------------------
# Merge per-split chunk files
# -----------------------------------------
merge_chunks()

# -----------------------------------------
# ✅ Finalization step (separate train/dev outputs)
# -----------------------------------------
def finalize_train_dev():
    """
    Finalize train/dev embeddings independently.
    Keeps separate .npy and idmap.json files ready for ModelForPer_slim_GNN.
    """
    print("\n🎯 Finalization complete — separate train/dev files kept.")
    print("✅ Files ready for model usage:")
    print(f"  ./bge_emb/task_{idx}_train_bge.npy")
    print(f"  ./bge_emb/task_{idx}_dev_bge.npy")
    print(f"  ./bge_emb/task_{idx}_train_bge_idmap.json")
    print(f"  ./bge_emb/task_{idx}_dev_bge_idmap.json")
    print("----------------------------------------------------------")

if __name__ == "__main__":
    finalize_train_dev()