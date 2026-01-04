
"""
Purpose:
--------
This test script is designed to **actually evaluate personalization effects** 
in the PersonaLLM_Slim_GNN model by using *real users* (from raw LaMP data) 
who have multiple queries.

Why this is needed:
-------------------
The original test_query_single.py only tested "history perturbation" within 
a SINGLE example (replace with random history / empty history). That cannot 
really measure user-level personalization.

Here:
  1. We explicitly get `user_id` from the raw dev set.
  2. Pick two or more *different* queries for the same user (requires ≥2 queries per user).
  3. Compare model outputs across three persona scenarios:
        a) Real persona → user's actual profile embeddings
        b) Other-user persona → profile embeddings from a completely different user
        c) Empty persona → no profile embeddings at all
  4. Measure cosine similarity between embeddings in these scenarios.
  5. Check effect on generated outputs.

Expected outcome:
-----------------
If personalization is working:
    * Real persona outputs will differ from those with "other-user" or "empty" profiles.
    * Embeddings for real vs other-user should have lower cosine similarity than real vs real (ideal case).

Notes:
------
- Works with task_id=3 data.
- Requires both `dev_questions.json` (raw) and `dev_profile.json` (aggregated with his_id) to exist.
"""

import os
import sys
import json
import random
import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoModel

# Allow imports from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from extention.ModelForPer_slim_GNN import PersonalLLM_Slim



# === Config toggles ===
USE_GATE = True
CHEKPOINT_NUM = 13750  # model checkpoint to load

# === Load tokenizers for LLM and embedding model ===
llm_tokenizer = AutoTokenizer.from_pretrained("../FlanT5-Large", use_fast=False)
emb_tokenizer = AutoTokenizer.from_pretrained("../bge-base-en-v1.5")

# === Load base LLM and fine-tuned weights ===
llm_model = T5ForConditionalGeneration.from_pretrained("../FlanT5-Large")
ckpt_path = f"../extention/output_3/checkpoint-{CHEKPOINT_NUM}/pytorch_model.bin"
state_dict = torch.load(ckpt_path, map_location="cpu")
llm_model.load_state_dict(state_dict, strict=False)

# === Load embedding model ===
emb_model = AutoModel.from_pretrained("../bge-base-en-v1.5")

## === Build persona‑aware model wrapper ===
model = PersonalLLM_Slim(
    llm_model=llm_model,
    emb_model=emb_model,
    max_input_len=256,
    max_new_len=32,
    task_id=3
)
model.eval()

# === Utility functions ===
def compute_profile_embs(his_id, emb_input):
    """Retrieve profile embeddings for history IDs; 
       make empty persona raw 768-D zeros before align_mlp."""
    with torch.no_grad():
        task_embs = model.obtain_task_emb(
            emb_input_ids=emb_input["input_ids"],
            emb_attention_mask=emb_input["attention_mask"],
            emb_token_type_ids=torch.zeros_like(emb_input["input_ids"])
        )

        if his_id.eq(0).all():
            if hasattr(model, "use_align_mlp") and model.use_align_mlp:
                # Construct raw BGE-space zero vector (768-D) and project
                neutral_vec = torch.zeros((1, model.emb_emb_size), 
                                          device=task_embs.device, dtype=task_embs.dtype)
                neutral_vec = model.align_mlp(neutral_vec)
                return neutral_vec
            else:
                # If no alignment layer, just return LLM-space zero vector
                return torch.zeros_like(task_embs)

        return model.obtain_profile_emb(his_id, task_embs)

def run_case(his_id, llm_input, emb_input):
    """Run model forward pass for a given persona history (`his_id`), returning decoded output text."""
    with torch.no_grad():
        _, seqs = model.forward(
            llm_input_ids=llm_input["input_ids"],
            llm_attention_mask=llm_input["attention_mask"],
            labels=torch.zeros_like(llm_input["input_ids"]),  # dummy labels
            emb_input_ids=emb_input["input_ids"],
            emb_attention_mask=emb_input["attention_mask"],
            emb_token_type_ids=torch.zeros_like(emb_input["input_ids"]),
            his_id=his_id
        )
    return llm_tokenizer.decode(seqs[0], skip_special_tokens=True).strip()

# === Load dev profiles (aggregated his_id) & raw dev questions (with user_id) ===

# 1️⃣ Choose dataset paths — full or subset
USE_SUBSET = True  # Set True to force using the _subset_id profiles + matching questions

if USE_SUBSET:
    profiles_path = "../LaMP_time_3_subset_id/dev_profile.json"  # his_id mapping file (subset)
    questions_path = "../LaMP_time_3_subset/dev_questions.json"  # matched subset questions, if exists
else:
    profiles_path = "../LaMP_time_3_id/dev_profile.json"  # his_id mapping file (full)
    questions_path = "../LaMP_time_3/dev_questions.json"  # raw full questions

# 2️⃣ Load data
if not os.path.exists(profiles_path):
    raise FileNotFoundError(f"Profiles file not found: {profiles_path}")
if not os.path.exists(questions_path):
    raise FileNotFoundError(f"Questions file not found: {questions_path}")

profiles = [json.loads(l) for l in open(profiles_path)]
raw_questions = json.load(open(questions_path))

# 3️⃣ Map from question.id → profile entry (restrict raw_questions to those in profiles)
id_to_profile = {p["id"]: p for p in profiles}
valid_ids = set(id_to_profile.keys())
raw_questions = [q for q in raw_questions if q["id"] in valid_ids]

# Group raw questions by user_id to find users with multiple examples
user_to_qids = {}
for q in raw_questions:
    uid = q["user_id"]
    user_to_qids.setdefault(uid, []).append(q["id"])

# Minimum queries per user required to include them in test
min_required_queries = 3  # set to 2 if requiring true multi-query comparison
user_to_qids = {u: ids for u, ids in user_to_qids.items() if len(ids) >= min_required_queries}

# === Main personalization sensitivity test ===
sample_size = min(5, len(user_to_qids))
print(f"\n[INFO] Found {len(user_to_qids)} users with ≥{min_required_queries} queries. Sampling {sample_size} users...\n")
for user_id, qids in random.sample(list(user_to_qids.items()), sample_size):
    print(f"\n=== USER {user_id} ===")

    # Filter out any qids not in the profile mapping (should not happen now)
    qids = [qid for qid in qids if qid in id_to_profile]
    if not qids:
        print(f"Skipping user {user_id} — no matching profiles in {profiles_path}")
        continue

    # Take first query for this user (the one we will test on)
    own_qid = qids[0]
    # Second query only if available
    other_qid_same_user = qids[1] if len(qids) > 1 else None

    # Load the real profile entry for this question
    own_profile = id_to_profile[own_qid]

    # Load actual question text for this query
    raw_q_entry = next(q for q in raw_questions if q["id"] == own_qid)
    query_text = raw_q_entry["input"]

    # Tokenize the task text for LLM and embedding model
    llm_input = llm_tokenizer(query_text, return_tensors="pt", max_length=256, truncation=True)
    emb_input = emb_tokenizer(query_text, return_tensors="pt", max_length=256, truncation=True)

    # Build real embedding
    his_id_real = torch.tensor(own_profile["his_id"]).unsqueeze(0)
    emb_real = compute_profile_embs(his_id_real, emb_input)

    # Pick maximally different 'other' user
    max_diff = -1
    best_other_profile = None
    for other_uid, qids_ in user_to_qids.items():
        if other_uid == user_id:
            continue
        candidate_profile = id_to_profile[qids_[0]]
        cand_emb = compute_profile_embs(torch.tensor(candidate_profile["his_id"]).unsqueeze(0), emb_input)
        diff_score = 1 - torch.nn.functional.cosine_similarity(emb_real, cand_emb).item()
        if diff_score > max_diff:
            max_diff = diff_score
            best_other_profile = candidate_profile
    his_id_other = torch.tensor(best_other_profile["his_id"]).unsqueeze(0)

    # Empty persona
    his_id_empty = torch.zeros_like(his_id_real)

    # Recompute for stats
    emb_other = compute_profile_embs(his_id_other, emb_input)
    emb_empty = compute_profile_embs(his_id_empty, emb_input)

    # Print similarities
    print(f"real_vs_other: cos={F.cosine_similarity(emb_real, emb_other).item():.4f}")
    print(f"real_vs_empty: cos={F.cosine_similarity(emb_real, emb_empty).item():.4f}")

    # Run cases with gate logging
    def run_with_gate(hid, label):
        with torch.no_grad():
            # Capture gate activation
            task_embs = model.obtain_task_emb(
                emb_input_ids=emb_input["input_ids"],
                emb_attention_mask=emb_input["attention_mask"],
                emb_token_type_ids=torch.zeros_like(emb_input["input_ids"])
            )
            profile_embs = compute_profile_embs(hid, emb_input)
            gate_input = torch.cat([task_embs, profile_embs], dim=-1)
            gate_val = torch.sigmoid(model.gate(gate_input)).mean().item()
            _, seqs = model.forward(
                llm_input_ids=llm_input["input_ids"],
                llm_attention_mask=llm_input["attention_mask"],
                labels=torch.zeros_like(llm_input["input_ids"]),
                emb_input_ids=emb_input["input_ids"],
                emb_attention_mask=emb_input["attention_mask"],
                emb_token_type_ids=torch.zeros_like(emb_input["input_ids"]),
                his_id=hid
            )
        out = llm_tokenizer.decode(seqs[0], skip_special_tokens=True).strip()
        print(f"{label} gate_avg={gate_val:.4f} → {out}")

    run_with_gate(his_id_real, "Real persona")
    run_with_gate(his_id_other, "Other persona")
    run_with_gate(his_id_empty, "Empty persona")
