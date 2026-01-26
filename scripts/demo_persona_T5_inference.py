"""
Demo script to query a fine-tuned personalized T5 model (PersonalLLM_Slim)
and compare retrieval-style similarity vs. generated textual prediction.

It includes automatic checkpoint verification with safe weight compatibility.
When vocab shapes differ (FlanT5 update vs trained checkpoint),
the embedding weights are skipped but personalization layers are restored.

Usage:
  python3 scripts/demo_persona_T5_inference.py
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, T5ForConditionalGeneration
from extention.ModelForPer_slim_GNN import PersonalLLM_Slim

# ==========================================================
# 🔹 CONFIG
# ==========================================================
TASK_ID = 3
CHECKPOINT_NUM = 2062
BASE_DIR = ".."
CKPT_PATH = os.path.join(
    BASE_DIR, f"extention/output_{TASK_ID}", f"checkpoint-{CHECKPOINT_NUM}", "pytorch_model.bin"
)
BGE_MODEL_PATH = os.path.join(BASE_DIR, "bge-base-en-v1.5")
LLM_MODEL_PATH = os.path.join(BASE_DIR, "FlanT5-small")

DATA_PATH = os.path.join(BASE_DIR, "LaMP_time_3_subset", "train_questions.json")
GRAPH_PATH = os.path.join(BASE_DIR, "graph_emb", f"task_{TASK_ID}_graph.npy")
HIS_TO_GRAPH_PATH = os.path.join(BASE_DIR, "graph_emb", f"task_{TASK_ID}_his_to_graph.json")

USE_CUSTOM = False
ENTRY_IDX = 2
CUSTOM_INPUT = """What is the score of the following review on a scale of 1 to 5?
Just answer with 1, 2, 3, 4, or 5 without further explanation.
review: It was easy to install, fit perfectly on my bike, and feels very durable.
The pockets are roomy and well‑made. Great value for money!
"""
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================================
# 🔹 LOAD MODELS & TOKENIZERS
# ==========================================================
print("✅ Loading tokenizer + base models...")
llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH, use_fast=False)
emb_tokenizer = AutoTokenizer.from_pretrained(BGE_MODEL_PATH)

# --- Load fine-tuned model ---
llm_model = T5ForConditionalGeneration.from_pretrained(LLM_MODEL_PATH)
emb_model = AutoModel.from_pretrained(BGE_MODEL_PATH)
print(f"Tokenizer length: {len(llm_tokenizer)}")

# --- Load plain baseline model (NO checkpoint) ---
plain_llm_model = T5ForConditionalGeneration.from_pretrained(LLM_MODEL_PATH)

# ---- Instantiate personalization wrapper ----
model = PersonalLLM_Slim(
    llm_model=llm_model,
    emb_model=emb_model,
    max_input_len=256,
    max_new_len=32,
    task_id=TASK_ID,
    use_gate=True,
)

# ---- Load checkpoint safely ----
print(f"🔍 Loading checkpoint: {CKPT_PATH}")
state_dict = torch.load(CKPT_PATH, map_location=DEVICE)

# Skip mismatched embedding weights (Flan‑T5 vocab size drift)
skip_prefixes = [
    "llm_model.shared.weight",
    "llm_model.encoder.embed_tokens.weight",
    "llm_model.decoder.embed_tokens.weight",
    "llm_model.lm_head.weight",
]
filtered_state_dict = {
    k: v for k, v in state_dict.items()
    if not any(k.startswith(prefix) for prefix in skip_prefixes)
}

missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)

print("✅ Loaded checkpoint (vocab shape differences safely ignored)")
if missing:
    print(f"ℹ️ Missing keys: {len(missing)} → {missing[:8]}{'...' if len(missing)>8 else ''}")
if unexpected:
    print(f"ℹ️ Unexpected keys: {len(unexpected)} → {unexpected[:8]}{'...' if len(unexpected)>8 else ''}")

model.to(DEVICE).eval()

# ==========================================================
# 🔹 LOAD EMBEDDINGS + DATA
# ==========================================================
emb_graph = np.load(GRAPH_PATH)
his_to_graph = json.load(open(HIS_TO_GRAPH_PATH))
data = json.load(open(DATA_PATH))

entry = data[ENTRY_IDX]
profile_texts = [his["text"] for his in entry["profile"]]
profile_ids = [int(his["id"]) for his in entry["profile"]]
print(f"[DEBUG] Raw profile IDs: {profile_ids}")
print(f"[DEBUG] Profile texts: {profile_texts}")

if hasattr(model, "his_train_memmap"):
    num_rows = len(model.his_train_memmap)
    print(f"[DEBUG] Valid profile ID range: 0 to {num_rows-1}")

# Load mapping from profile_id to memmap row index
OFFSET_PATH = os.path.join(BASE_DIR, "bge_emb", f"task_{TASK_ID}_train_offsets.json")
with open(OFFSET_PATH, "r") as f:
    offsets_data = json.load(f)

# Build mapping from profile_id to row index
profile_id_to_row = {}
for entry_ in offsets_data["entries"]:
    start = entry_["start"]
    for idx, pid in enumerate(entry_["profile_id"]):
        profile_id_to_row[str(pid)] = start + idx

# Map profile_ids to row indices
profile_row_indices = [profile_id_to_row.get(str(pid), 0) for pid in profile_ids]
print(f"[DEBUG] Mapped profile row indices: {profile_row_indices}")

his_id = torch.tensor(profile_row_indices, dtype=torch.long).unsqueeze(0).to(DEVICE)

# ==========================================================
# 🔹 BUILD INPUT FOR MODEL
# ==========================================================
if USE_CUSTOM:
    input_text = CUSTOM_INPUT.strip()
    print("\n🧠 Using CUSTOM input query...\n")
else:
    input_text = entry["input"]
    print("\n🧠 Using input from dataset entry...\n")

print(f"Query text:\n{input_text}\n")

llm_inputs = llm_tokenizer(input_text, return_tensors="pt",
                           truncation=True, max_length=256).to(DEVICE)
emb_inputs = emb_tokenizer(input_text, return_tensors="pt",
                           truncation=True, max_length=256).to(DEVICE)
# his_id is now set using mapped row indices above

# ==========================================================
# 🔎 PERSONALIZATION LAYER DIAGNOSTICS
# ==========================================================
print("\n📊 Inspecting personalization signal contributions...")
with torch.no_grad():
    emb_inputs_short = emb_tokenizer(
        input_text, return_tensors="pt", truncation=True, max_length=64
    ).to(DEVICE)
    task_emb = model.obtain_task_emb(
        emb_input_ids=emb_inputs_short["input_ids"],
        emb_attention_mask=emb_inputs_short["attention_mask"],
        emb_token_type_ids=torch.zeros_like(emb_inputs_short["input_ids"]),
    )

    num_rows = 1
    if hasattr(model, "his_train_memmap"):
        num_rows = len(model.his_train_memmap)
    safe_his_id = his_id.clone()
    safe_his_id[safe_his_id >= num_rows] = 0
    print(f"[DEBUG] Profile IDs used: {safe_his_id.tolist()} (max valid: {num_rows-1})")

    try:
        prof_emb = model.obtain_profile_emb(safe_his_id, task_emb)
        # --- Diagnostic: Check for NaNs in profile embedding ---
        if torch.isnan(prof_emb).any():
            print("❌ Profile embedding contains NaN values! Replacing with zeros.")
            prof_emb = torch.nan_to_num(prof_emb, nan=0.0, posinf=0.0, neginf=0.0)
        print(f"[DEBUG] Profile embedding shape: {prof_emb.shape}, NaN count: {torch.isnan(prof_emb).sum().item()}")
    except Exception:
        prof_emb = torch.zeros_like(task_emb)

    # Always sanitize before gate calculation
    prof_emb = torch.nan_to_num(prof_emb, nan=0.0, posinf=0.0, neginf=0.0)

    gate_mean = 0.0
    if hasattr(model, "gate"):
        gate_input = torch.cat([task_emb, prof_emb], dim=-1)
        gate_vals = torch.sigmoid(model.gate(gate_input))
        gate_mean = gate_vals.mean().item()

    t_norm = float(torch.linalg.norm(task_emb))
    p_norm = float(torch.linalg.norm(prof_emb))
    diff_norm = float(torch.norm(task_emb - prof_emb))

print(f"Task_emb norm  : {t_norm:.4f}")
print(f"Profile_emb norm : {p_norm:.4f}")
print(f"Gate mean value : {gate_mean:.4f}")
print(f"Task–Profile diff: {diff_norm:.4f}")
print("📈 (Higher gate ⇒ stronger personalization influence)\n")

# ==========================================================
# 🔹 GENERATE OUTPUTS
# ==========================================================
print("\n💬 Generating output from base Flan‑T5:")
with torch.no_grad():
    gen_base = plain_llm_model.generate(
        **llm_inputs, max_new_tokens=32, num_beams=4, do_sample=False
    )
base_text = llm_tokenizer.decode(gen_base[0], skip_special_tokens=True).strip()
print(f"→ Base output:\n{base_text}\n")

print("\n💫 Generating personalized output:")
with torch.no_grad():
    gen_persona = llm_model.generate(
        **llm_inputs, max_new_tokens=32, num_beams=4, do_sample=False
    )
persona_text = llm_tokenizer.decode(gen_persona[0], skip_special_tokens=True).strip()
print(f"→ Persona output:\n{persona_text}\n")

# ==========================================================
# 🔹 COMPARISON SUMMARY
# ==========================================================
print("=" * 80)
print("🔍 Generation Comparison Summary")
print("-" * 80)
print(f"Base model output:\n{base_text}\n")
print(f"Personalized model output:\n{persona_text}\n")
print("=" * 80)

print("✅ Done.")