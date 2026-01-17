"""
Purpose:
--------
Evaluate personalization effects of PersonaLLM_Slim_GNN model by comparing
generation using real persona vs empty persona.

This version:
- Merges dev_questions.json and dev_outputs.json for gold labels.
- Resolves persona ID mapping using ./bge_emb/task_3_dev_bge_idmap.json.
- Avoids IndexError from raw LaMP review IDs during memmap embedding lookup.
"""

import os
import sys
import json
import random
import re
import math
import torch
import numpy as np
import torch.nn.functional as F
from collections import Counter
from sklearn.metrics import accuracy_score, mean_squared_error
from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoModel

# Allow imports from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from extention.ModelForPer_slim_GNN import PersonalLLM_Slim

# === Config toggles ===
USE_GATE = True
CHECKPOINT_NUM = 687
TASK_ID = 3
USE_SUBSET = True

# === Tokenizers ===
llm_tokenizer = AutoTokenizer.from_pretrained("../FlanT5-small", use_fast=False)
emb_tokenizer = AutoTokenizer.from_pretrained("../bge-base-en-v1.5")

# === Load model & checkpoint ===
llm_model = T5ForConditionalGeneration.from_pretrained("../FlanT5-small")
ckpt_path = f"../extention/output_{TASK_ID}/checkpoint-{CHECKPOINT_NUM}/pytorch_model.bin"
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"❌ Missing checkpoint: {ckpt_path}")
state_dict = torch.load(ckpt_path, map_location="cpu")
llm_model.load_state_dict(state_dict, strict=False)

emb_model = AutoModel.from_pretrained("../bge-base-en-v1.5")

# === Instantiate personalization model ===
model = PersonalLLM_Slim(
    llm_model=llm_model,
    emb_model=emb_model,
    max_input_len=256,
    max_new_len=32,
    task_id=TASK_ID,
    use_gate=USE_GATE,
)
model.eval()
print(f"[DEBUG] LLM device: {next(llm_model.parameters()).device}")
print(f"[DEBUG] Embedding model device: {next(emb_model.parameters()).device}")
print(f"[DEBUG] Model params loaded: {len(state_dict)} tensors")

# --- Diagnostic and initialization block ---
try:
    train_shape = getattr(model, "his_train_memmap", None)
    dev_shape = getattr(model, "his_dev_memmap", None)
    print(f"[DEBUG] Memmap train/dev shapes:",
          train_shape.shape if train_shape is not None else "missing",
          dev_shape.shape if dev_shape is not None else "missing")
except Exception as e:
    print(f"[WARN] Could not read memmap shapes — {e}")

# Ensure memmap actually contains real non‑zero embeddings
if hasattr(model, "his_dev_memmap"):
    sample_rows = np.array(model.his_dev_memmap[:5])
    zeros_ratio = np.mean(np.isclose(sample_rows, 0))
    print(f"[DEBUG] First 5 embedding rows — zeros ratio: {zeros_ratio:.4f}")

# --- Gate sanity check (avoid NaN) ---
if hasattr(model, "gate"):
    gate_weight_nan = torch.isnan(model.gate.weight).any().item()
    gate_bias_nan = torch.isnan(model.gate.bias).any().item()
    if gate_weight_nan or gate_bias_nan:
        print("[WARN] Detected NaN gate weights — reinitializing gate layer.")
        torch.nn.init.zeros_(model.gate.weight)
        torch.nn.init.zeros_(model.gate.bias)
    else:
        print("[DEBUG] Gate weights look normal.")
else:
    print("[INFO] Model has no gate layer detected.")

# === Utility functions ===
def compute_profile_embs(his_id, emb_input):
    """Compute profile embeddings safely."""
    with torch.no_grad():
        task_embs = model.obtain_task_emb(
            emb_input_ids=emb_input["input_ids"],
            emb_attention_mask=emb_input["attention_mask"],
            emb_token_type_ids=torch.zeros_like(emb_input["input_ids"])
        )

        # === Empty persona fallback ===
        if his_id.eq(0).all():
            neutral_vec = torch.zeros((1, model.emb_emb_size), device=task_embs.device)
            neutral_vec = model.align_mlp(neutral_vec)
            return neutral_vec

        # === Obtain real persona embedding ===
        emb_vec = model.obtain_profile_emb(his_id, task_embs)
        emb_vec = torch.nan_to_num(emb_vec, nan=0.0, posinf=0.0, neginf=0.0)

        # --- Normalization + dropout noise ---
        # --- Strengthened normalization + adaptive noise ---
        emb_vec = F.normalize(emb_vec, p=2, dim=-1)
        # add magnitudes proportional to task embedding norm
        scale = task_embs.norm(p=2, dim=-1, keepdim=True) * 0.05
        noise = torch.randn_like(emb_vec) * scale
        emb_vec = emb_vec + noise
        emb_vec = F.normalize(emb_vec, p=2, dim=-1)

        return emb_vec

def run_case(llm_input, emb_input, his_id):
    """Run forward generation for a given persona tensor with gate diagnostics."""
    with torch.no_grad():
        # Task embeddings
        task_embs = model.obtain_task_emb(
            emb_input_ids=emb_input["input_ids"],
            emb_attention_mask=emb_input["attention_mask"],
            emb_token_type_ids=torch.zeros_like(emb_input["input_ids"])
        )
        task_embs = torch.nan_to_num(task_embs, nan=0.0, posinf=0.0, neginf=0.0)

        # Persona embeddings
        persona_embs = model.obtain_profile_emb(his_id, task_embs)
        persona_embs = torch.nan_to_num(persona_embs, nan=0.0, posinf=0.0, neginf=0.0)

        # ---- GATE MONITOR ----
        if hasattr(model, "gate"):
            gate_in = torch.cat([task_embs, persona_embs], dim=-1)
            gate_in = torch.nan_to_num(gate_in, nan=0.0, posinf=0.0, neginf=0.0)
            gate_out = torch.sigmoid(model.gate(gate_in))
            # Compute statistics safely
            mean_val = gate_out.mean().item()
            std_val = float(gate_out.std().item()) if gate_out.numel() > 1 else 0.0
            if math.isfinite(std_val):
                print(f"[Gate‑Monitor] mean={mean_val:.4f} ±{std_val:.4f}")
            else:
                print(f"[Gate‑Monitor] mean={mean_val:.4f} ±0.0000 (flat activation)")
                gate_out = torch.nan_to_num(gate_out, nan=0.0, posinf=0.0, neginf=0.0)
        # ----------------------

        # Forward generation
        # --- PATCH: Handle models that expect labels to always be a tensor ---
        # If model.forward fails with labels=None, fallback to dummy tensor.
        try:
            _, seqs = model.forward(
                llm_input_ids=llm_input["input_ids"],
                llm_attention_mask=llm_input["attention_mask"],
                labels=None,  # Preferred: None for inference
                emb_input_ids=emb_input["input_ids"],
                emb_attention_mask=emb_input["attention_mask"],
                emb_token_type_ids=torch.zeros_like(emb_input["input_ids"]),
                his_id=his_id
            )
        except AttributeError as e:
            # Fallback: pass a dummy tensor if labels=None causes error
            if "'NoneType' object has no attribute 'long'" in str(e):
                dummy_labels = torch.zeros_like(llm_input["input_ids"])
                _, seqs = model.forward(
                    llm_input_ids=llm_input["input_ids"],
                    llm_attention_mask=llm_input["attention_mask"],
                    labels=dummy_labels,
                    emb_input_ids=emb_input["input_ids"],
                    emb_attention_mask=emb_input["attention_mask"],
                    emb_token_type_ids=torch.zeros_like(emb_input["input_ids"]),
                    his_id=his_id
                )
            else:
                raise
        output_text = llm_tokenizer.decode(seqs[0], skip_special_tokens=True).strip()
        if output_text == "" or set(output_text) == {"."}:
            output_text = "(no meaningful output)"
        return output_text

# === Load combined question & gold output files ===
if USE_SUBSET:
    questions_path = "../LaMP_time_3_subset/dev_questions.json"
    outputs_path   = "../LaMP_time_3_subset/dev_outputs.json"
else:
    questions_path = "../LaMP_time_3/dev_questions.json"
    outputs_path   = "../LaMP_time_3/dev_outputs.json"

if not (os.path.exists(questions_path) and os.path.exists(outputs_path)):
    raise FileNotFoundError(f"❌ Missing dev data: {questions_path} or {outputs_path}")

with open(questions_path, "r") as fq:
    questions_data = json.load(fq)
with open(outputs_path, "r") as fo:
    outputs_data = json.load(fo)

# Build mapping of gold labels
id_to_gold = {item["id"]: item["output"] for item in outputs_data.get("golds", [])}

print(f"[INFO] Loaded {len(questions_data)} question entries and {len(id_to_gold)} gold outputs.")

# Merge profiles & golds
id_to_profile = {}
raw_questions = []
for entry in questions_data:
    qid = entry.get("id")
    uid = entry.get("user_id")
    profile = entry.get("profile", [])
    qtext = entry.get("input", "")
    gold_output = str(id_to_gold.get(qid, "")).strip()
    his_ids = [it.get("id") for it in profile if "id" in it]
    norm_entry = {
        "id": qid,
        "user_id": uid,
        "input": qtext,
        "profile": profile,
        "his_id": his_ids,
        "gold_output": gold_output,
    }
    id_to_profile[qid] = norm_entry
    raw_questions.append(norm_entry)

print(f"[INFO] Profiles merged with gold outputs: {len(raw_questions)} total entries")

print(f"[INFO] Profile dictionary size: {len(id_to_profile)}")
print(f"[INFO] Question entries: {len(raw_questions)}")

# === Analyze user coverage ===
user_counts = Counter(q.get("user_id") for q in raw_questions if q.get("user_id"))
print(f"[DEBUG] Unique users: {len(user_counts)}")
print(f"[DEBUG] Top 10 user question counts: {user_counts.most_common(10)}")

# === Group valid users with sufficient reviews ===
min_required_reviews = 3
user_to_qids = {}
for u in user_counts.keys():
    ids = [q["id"] for q in raw_questions if q.get("user_id") == u]
    if len(id_to_profile.get(ids[0], {}).get("profile", [])) >= min_required_reviews:
        user_to_qids[u] = ids

print(f"[INFO] Valid users: {len(user_to_qids)} found with ≥{min_required_reviews} profile reviews.")
sample_size = min(5, len(user_to_qids))
avg_reviews = np.mean([len(id_to_profile.get(ids[0], {}).get("profile", [])) for ids in user_to_qids.values()]) if user_to_qids else 0
print(f"[INFO] Average profile review count among selected users: {avg_reviews:.2f}")

# === Helper for numeric parsing ===
def extract_numeric_rating(text):
    match = re.search(r"\b([1-5])(\.0+)?\b", str(text))
    return int(match.group(1)) if match else None

# === Load ID→row mapping for embeddings ===
map_path = "../bge_emb/task_3_dev_bge_idmap.json"
if not os.path.exists(map_path):
    raise FileNotFoundError(f"❌ Missing ID map file: {map_path}")
with open(map_path, "r") as f:
    his_id_to_row = json.load(f)

print(f"[INFO] Loaded ID→row mapping with {len(his_id_to_row)} entries.")

# === Evaluation ===
results = []
correct_count = 0
total_samples = 0

for user_id, qids in random.sample(list(user_to_qids.items()), sample_size):
    print(f"\n=== USER {user_id} ===")
    qids = [qid for qid in qids if qid in id_to_profile]
    if not qids:
        continue
    own_qid = qids[0]
    raw_q_entry = id_to_profile[own_qid]
    query_text = raw_q_entry["input"]
    gold_output = raw_q_entry["gold_output"]

    llm_input = llm_tokenizer(query_text, return_tensors="pt", truncation=True, max_length=256)
    emb_input = emb_tokenizer(query_text, return_tensors="pt", truncation=True, max_length=256)

    # === Map history IDs to memmap row indices ===
    safe_his_ids = []
    for hid in raw_q_entry["his_id"]:
        try:
            row_idx = his_id_to_row.get(str(hid), 0)
            safe_his_ids.append(int(row_idx))
        except Exception:
            safe_his_ids.append(0)

    his_id_real = torch.tensor(safe_his_ids, dtype=torch.long).unsqueeze(0)
    his_id_empty = torch.zeros_like(his_id_real, dtype=torch.long)

    emb_real = compute_profile_embs(his_id_real, emb_input)
    emb_empty = compute_profile_embs(his_id_empty, emb_input)
    print(f"real_vs_empty_cos = {F.cosine_similarity(emb_real, emb_empty).item():.4f}")

    # === Model predictions ===
    pred_real = run_case(llm_input, emb_input, his_id_real)
    pred_empty = run_case(llm_input, emb_input, his_id_empty)

    print(f"Gold: {gold_output} | Real persona → {pred_real} | Empty persona → {pred_empty}")

    gold_num = extract_numeric_rating(gold_output)
    pred_real_num = extract_numeric_rating(pred_real)
    pred_empty_num = extract_numeric_rating(pred_empty)

    results.append({
        "user_id": user_id,
        "qid": own_qid,
        "gold": gold_output,
        "pred_real": pred_real,
        "pred_empty": pred_empty,
        "gold_num": gold_num,
        "pred_real_num": pred_real_num,
        "pred_empty_num": pred_empty_num
    })

    if pred_real.strip() == gold_output.strip():
        correct_count += 1
    total_samples += 1

# === Compute metrics ===
text_acc = correct_count / total_samples if total_samples else 0
gold_numeric = [r["gold_num"] for r in results]
real_numeric = [r["pred_real_num"] for r in results]
empty_numeric = [r["pred_empty_num"] for r in results]

acc_real = acc_empty = rmse_real = rmse_empty = None
valid_real = [(g, p) for g, p in zip(gold_numeric, real_numeric) if g is not None and p is not None]
valid_empty = [(g, p) for g, p in zip(gold_numeric, empty_numeric) if g is not None and p is not None]

if valid_real:
    g, p = zip(*valid_real)
    acc_real = accuracy_score(g, p)
    rmse_real = mean_squared_error(g, p, squared=False)
if valid_empty:
    g2, p2 = zip(*valid_empty)
    acc_empty = accuracy_score(g2, p2)
    rmse_empty = mean_squared_error(g2, p2, squared=False)

# === Display personalization metrics ===
print("\n=== PERSONALIZATION METRICS ===")
print(f"Textual match accuracy: {text_acc:.3f}")
if acc_real is not None and acc_empty is not None:
    print(f"Numeric accuracy → With persona: {acc_real:.3f} | Without persona: {acc_empty:.3f}")
if rmse_real is not None and rmse_empty is not None:
    print(f"Numeric RMSE → With persona: {rmse_real:.3f} | Without persona: {rmse_empty:.3f}")