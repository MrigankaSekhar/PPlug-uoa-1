"""
LaMP Task 3 (Review Scoring) — Slim Embedding Script (Mac/MPS-friendly) + Metrics

Adds post-embedding diagnostics:
  • Cosine similarity stats among a random sample of REVIEW vectors (mean/std/min/max, histogram).
  • Percentage of reviews that hit tokenizer truncation (length >= max_length after tokenization).
  • Saves metrics to JSON next to outputs.

Behavior (matches upstream PPlug embedding for idx == 3):
  • Input question: extract ONLY the substring after "review:" (case-insensitive).
  • Profile corpus: profile[*]["text"].
  • Embeddings: CLS token of last_hidden_state + L2 normalization.
  • Concatenation order per entry: [profile_0, ..., profile_n, question].
  • Streams embeddings to a NumPy memmap + writes offsets.json.

Usage:
  python3 embedding_lamp3_mmap_metrics.py     --dataset-root .     --model ./bge-base-en-v1.5     --out-dir ./bge_emb     --splits train dev     --batch-size 128     --max-length 512     --device auto     --metrics-sample 1000
"""

import argparse
import json
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import re
import random
from typing import Dict, List, Tuple, Iterable

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm  # <-- Add tqdm import

# -----------------------------
# Device selection
# -----------------------------

def pick_device(prefer: str = "auto") -> torch.device:
    prefer = (prefer or "auto").lower()
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if prefer == "cpu":
        return torch.device("cpu")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# -----------------------------
# Task 3 shaping
# -----------------------------

_REVIEW_RE = re.compile(r"review:\s*(.*)$", re.IGNORECASE | re.DOTALL)

def process_entry_lamp3(entry: Dict) -> Dict[str, List[str]]:
    """
    Returns dict with:
      question    -> str (only text after 'review:')
      profile_txt -> list[str] (his['text'])
      profile_id  -> list[str] (his['id'])
    """
    question = entry.get("input", "") or ""
    m = _REVIEW_RE.search(question)
    real_question = m.group(1).strip() if m else question.strip()

    profile = entry.get("profile", []) or []
    profile_txt = []
    profile_id = []
    for his in profile:
        profile_txt.append(str(his.get("text", "")))
        profile_id.append(str(his.get("id", "")))

    return {
        "question": real_question,
        "profile_txt": profile_txt,
        "profile_id": profile_id,
    }

# -----------------------------
# Embedding (CLS + L2)
# -----------------------------

def embed_texts(
    model,
    tokenizer,
    texts: List[str],
    device: torch.device,
    batch_size: int = 128,
    max_length: int = 512,
) -> Iterable[torch.Tensor]:
    """Yield L2-normalized CLS embeddings in batches."""
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            cls = outputs[0][:, 0]
            emb = torch.nn.functional.normalize(cls, p=2, dim=1)
            yield emb.detach().cpu()

# -----------------------------
# Dataset iteration + memmap writer
# -----------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def count_texts_and_offsets(dataset: List[Dict]) -> Tuple[int, List[Dict]]:
    total = 0
    offsets = []
    for entry in dataset:
        shaped = process_entry_lamp3(entry)
        n = len(shaped["profile_txt"]) + 1  # + question
        offsets.append({
            "start": total,
            "n_profiles": n - 1,
            "question_index": total + (n - 1),
            "profile_id": shaped.get("profile_id", []),
        })
        total += n
    return total, offsets


def iterate_texts(dataset: List[Dict]) -> Iterable[str]:
    for entry in dataset:
        shaped = process_entry_lamp3(entry)
        for t in shaped["profile_txt"]:
            yield t
        yield shaped["question"]

# -----------------------------
# Metrics
# -----------------------------

def compute_review_truncation_rate(dataset: List[Dict], tokenizer, max_length: int) -> float:
    total = 0
    capped = 0
    for entry in dataset:
        shaped = process_entry_lamp3(entry)
        txt = shaped["question"]
        enc = tokenizer(
            txt,
            padding=False,
            truncation=True,
            max_length=max_length,
            return_tensors=None,
        )
        # Robustly get tokenized length
        if isinstance(enc, dict):
            length = len(enc["input_ids"])
        elif hasattr(enc, "ids"):
            length = len(enc.ids)
        elif isinstance(enc, list) and hasattr(enc[0], "ids"):
            length = len(enc[0].ids)
        else:
            length = 0
        total += 1
        if length >= max_length:
            capped += 1
    return (capped / total) if total > 0 else 0.0


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def sample_review_cosine_stats(mmap_path: str, offsets_path: str, sample_size: int = 1000, seed: int = 13):
    """Load review vectors from memmap via question_index and compute similarity stats on a random subset."""
    rng = random.Random(seed)

    with open(offsets_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    entries = meta["entries"]
    total_reviews = len(entries)

    if total_reviews == 0:
        return {
            "n_reviews": 0,
            "sampled": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "hist": None,
            "bins": None,
        }

    # Choose sample indices of entries
    idxs = list(range(total_reviews))
    rng.shuffle(idxs)
    idxs = idxs[: min(sample_size, total_reviews)]

    # Map to review vector row indices
    review_rows = [entries[i]["question_index"] for i in idxs]
    # Load memmap lazily
    mm = np.memmap(mmap_path, mode='r', dtype='float32', shape=(meta["total_vectors"], meta["dim"]))
    R = np.asarray([mm[row] for row in review_rows])  # (k, dim)

    # Pairwise cosine for upper triangle only to keep compute lighter on larger k
    # For k up to ~1k, this is manageable.
    k = R.shape[0]
    if k <= 1:
        return {
            "n_reviews": total_reviews,
            "sampled": k,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "hist": None,
            "bins": None,
        }

    # Normalize again defensively (should already be unit-norm)
    Rn = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-12)

    # Compute pairwise cosine similarities efficiently using matrix multiply
    # but keep only upper triangle without diagonal
    S = Rn @ Rn.T
    triu = np.triu_indices(k, k=1)
    sims = S[triu]

    stats = {
        "n_reviews": total_reviews,
        "sampled": int(k),
        "mean": float(np.mean(sims)),
        "std": float(np.std(sims)),
        "min": float(np.min(sims)),
        "max": float(np.max(sims)),
    }

    # Histogram for quick visual summary
    hist, bin_edges = np.histogram(sims, bins=20, range=(-0.2, 1.0))
    stats["hist"] = hist.tolist()
    stats["bins"] = [float(x) for x in bin_edges.tolist()]

    return stats

# -----------------------------
# Split runner
# -----------------------------

def run_split(
    split: str,
    dataset_root: str,
    out_dir: str,
    model,
    tokenizer,
    device: torch.device,
    batch_size: int,
    max_length: int,
    metrics_sample: int,
):
    dir_name = dataset_root
    file_name = os.path.join(dir_name, f"{split}_questions.json")
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"Missing dataset file: {file_name}")

    with open(file_name, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    total_texts, offsets = count_texts_and_offsets(dataset)
    print(f"[DEBUG] Split '{split}' has {total_texts} texts to embed.")
    print(f"[DEBUG] Number of entries: {len(dataset)}")
    print(f"[DEBUG] Example entry: {dataset[0] if len(dataset) > 0 else 'EMPTY'}")

    # Probe embedding dim
    probe_emb = next(embed_texts(model, tokenizer, ["test"], device, batch_size=1, max_length=max_length))
    dim = int(probe_emb.shape[1])

    # Prepare memmap
    ensure_dir(out_dir)
    mmap_path = os.path.join(out_dir, f"task_3_{split}_bge.memmap.npy")
    mmap = np.memmap(mmap_path, mode="w+", dtype="float32", shape=(total_texts, dim))

    # Stream write
    write_ptr = 0
    buffer: List[str] = []

    def flush(buf: List[str], ptr: int) -> int:
        if not buf:
            return ptr
        for emb_batch in embed_texts(model, tokenizer, buf, device, batch_size=batch_size, max_length=max_length):
            bsz = emb_batch.shape[0]
            mmap[ptr : ptr + bsz, :] = emb_batch.numpy()
            ptr += bsz
        buf.clear()
        return ptr

    # --- Wrap with tqdm ---
    texts_iter = iterate_texts(dataset)
    pbar = tqdm(total=total_texts, desc=f"Embedding {split}", unit="vec")

    for text in texts_iter:
        buffer.append(text)
        if len(buffer) >= batch_size:
            prev_ptr = write_ptr
            write_ptr = flush(buffer, write_ptr)
            pbar.update(write_ptr - prev_ptr)
    prev_ptr = write_ptr
    write_ptr = flush(buffer, write_ptr)
    pbar.update(write_ptr - prev_ptr)
    pbar.close()
    if write_ptr != total_texts:
        raise RuntimeError(f"Wrote {write_ptr} vectors but expected {total_texts}")

    mmap.flush()

    # Save offsets
    offsets_path = os.path.join(out_dir, f"task_3_{split}_offsets.json")
    with open(offsets_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_vectors": total_texts,
            "dim": dim,
            "entries": offsets,
        }, f, ensure_ascii=False, indent=2)

    print(f"[OK] {split}: {total_texts} vectors, dim={dim} -> {mmap_path}")

    # ------------------ Metrics ------------------
    # 1) Review truncation rate
    trunc_rate = compute_review_truncation_rate(dataset, tokenizer, max_length=max_length)

    # 2) Cosine similarity stats on review vectors
    cos_stats = sample_review_cosine_stats(
        mmap_path=mmap_path,
        offsets_path=offsets_path,
        sample_size=metrics_sample,
        seed=13,
    )

    metrics = {
        "split": split,
        "n_entries": len(dataset),
        "embedding_dim": dim,
        "review_truncation_rate": trunc_rate,  # fraction in [0,1]
        "review_cosine_stats": cos_stats,
    }

    metrics_path = os.path.join(out_dir, f"task_3_{split}_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("[METRICS]", json.dumps(metrics, indent=2))
    print(f"[OK] Metrics saved -> {metrics_path}")

# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="LaMP-3 (review scoring) embedding generator (memmap) + metrics")
    ap.add_argument("--dataset-root", type=str, default=".", help="Root folder containing LaMP_time_3/")
    ap.add_argument("--model", type=str, default="./bge-base-en-v1.5", help="HF model path or name")
    ap.add_argument("--out-dir", type=str, default="./bge_emb", help="Output directory")
    ap.add_argument("--splits", nargs="+", default=["train", "dev"], help="Splits to process")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--device", type=str, default="auto", help="cuda|mps|cpu|auto")
    ap.add_argument("--metrics-sample", type=int, default=1000, help="Number of review vectors to sample for cosine stats")

    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device)

    ensure_dir(args.out_dir)
    for split in args.splits:
        run_split(
            split=split,
            dataset_root=args.dataset_root,
            out_dir=args.out_dir,
            model=model,
            tokenizer=tokenizer,
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            metrics_sample=args.metrics_sample,
        )

if __name__ == "__main__":
    main()
