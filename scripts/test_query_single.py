
import torch
import json
import random
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoModel
from extention.ModelForPer_slim_GNN import PersonalLLM_Slim
import torch.nn.functional as F

# ======== summary ========
# Test script for querying a fine-tuned PersonalLLM_Slim model.
# Runs sample seen and unseen queries with:
# - real user profile embeddings
# - random generic user profile embeddings
# - empty user profile embeddings
# Prints outputs and profile embedding statistics.
# =======================

# === DEBUG TOGGLES ===
USE_PROFILE = True
USE_GATE = True
CHEKPOINT_NUM=69
# Tokenizers
llm_tokenizer = AutoTokenizer.from_pretrained("../FlanT5-small", use_fast=False)
emb_tokenizer = AutoTokenizer.from_pretrained("../bge-base-en-v1.5")

# Load base LLM + checkpoint
llm_model = T5ForConditionalGeneration.from_pretrained("../FlanT5-small")
ckpt_path = f"../extention/output_3/checkpoint-{CHEKPOINT_NUM}/pytorch_model.bin"
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"❌ Checkpoint not found at {ckpt_path}")
state_dict = torch.load(ckpt_path, map_location="cpu")
llm_model.load_state_dict(state_dict, strict=False)

# Embedding model
emb_model = AutoModel.from_pretrained("../bge-base-en-v1.5")

# Persona‑aware model
model = PersonalLLM_Slim(
    llm_model=llm_model,
    emb_model=emb_model,
    max_input_len=256,
    max_new_len=32,
    task_id=3,
    # cfg={"use_profile_emb": USE_PROFILE, "use_gate": USE_GATE}
)
model.eval()

# --- Auto–dataset path selection ---
profiles_path = "../LaMP_time_3_subset_id/dev_profile.json"
questions_path = "../LaMP_time_3/dev_questions.json"

if not os.path.exists(profiles_path):
    raise FileNotFoundError(f"❌ Profiles file not found: {profiles_path}")
if not os.path.exists(questions_path):
    # fall back to subset questions if it exists
    subset_q = "../LaMP_time_3_subset_id/dev_questions.json"
    if os.path.exists(subset_q):
        questions_path = subset_q
    else:
        raise FileNotFoundError(f"❌ Questions file not found: {questions_path}")

# Load dev profiles & questions
with open(profiles_path) as f:
    profiles = [json.loads(l) for l in f]
with open(questions_path) as fq:
    # Detect whether list-of-objs or JSON lines
    first_char = fq.read(1)
    fq.seek(0)
    if first_char.strip().startswith("["):
        dev_questions = json.load(fq)
    else:
        dev_questions = [json.loads(l) for l in fq]

def get_user_id(profile, idx):
    return profile.get("user_id") or profile.get("user") or f"profileIdx-{idx}"

def compute_profile_embs(his_id, emb_input):
    """Run model's own task→profile embedding path."""
    with torch.no_grad():
        task_embs = model.obtain_task_emb(
            emb_input_ids=emb_input["input_ids"],
            emb_attention_mask=emb_input["attention_mask"],
            emb_token_type_ids=torch.zeros_like(emb_input["input_ids"])
        )
        return model.obtain_profile_emb(his_id, task_embs)

def run_case(his_id, llm_input, emb_input):
    """Run inference + return decoded output."""
    with torch.no_grad():
        _, seqs = model.forward(
            llm_input_ids=llm_input["input_ids"],
            llm_attention_mask=llm_input["attention_mask"],
            labels=None,
            emb_input_ids=emb_input["input_ids"],
            emb_attention_mask=emb_input["attention_mask"],
            emb_token_type_ids=torch.zeros_like(emb_input["input_ids"]),
            his_id=his_id
        )
    return llm_tokenizer.decode(seqs[0], skip_special_tokens=True)

def persona_stats(real_emb, other_emb, other_label):
    cos = F.cosine_similarity(real_emb, other_emb).item()
    return f"{other_label}: cos={cos:.4f}, norm_real={real_emb.norm().item():.2f}, norm_{other_label}={other_emb.norm().item():.2f}"

# --- Query sets ---
unseen_queries = [
    ("UNSEEN", "What is the score of the following review on a scale of 1 to 5? just answer with 1, 2, 3, 4, or 5 without further explanation. review: The headphones arrived quickly and were packaged well. Sound quality is good for the price, although the ear pads feel a bit cheap.",
     random.randrange(len(profiles))),
    ("UNSEEN", "What is the score of the following review on a scale of 1 to 5? just answer with 1, 2, 3, 4, or 5 without further explanation. review: The book was fast-paced and entertaining, but some plot points were predictable. Still enjoyed it overall.",
     random.randrange(len(profiles)))
]

seen_queries = [("SEEN", q["input"], i) for i, q in enumerate(dev_questions[:2])]
test_queries = unseen_queries + seen_queries

# --- Main loop ---
all_ids_flat = [hid for p in profiles for hid in p["his_id"]]

for case_type, query, prof_idx in test_queries:
    user_ctx = profiles[prof_idx]
    real_user_id = get_user_id(user_ctx, prof_idx)
    his_id_real = torch.tensor(user_ctx["his_id"]).unsqueeze(0)
    llm_input = llm_tokenizer(query, return_tensors="pt", max_length=256, truncation=True)
    emb_input = emb_tokenizer(query, return_tensors="pt", max_length=256, truncation=True)

    # Embeddings
    profile_real = compute_profile_embs(his_id_real, emb_input)
    his_id_generic = torch.tensor(random.sample(all_ids_flat, len(user_ctx["his_id"]))).unsqueeze(0)
    profile_generic = compute_profile_embs(his_id_generic, emb_input)
    his_id_empty = torch.zeros_like(his_id_real)
    profile_empty = compute_profile_embs(his_id_empty, emb_input)

    # Outputs
    out_real = run_case(his_id_real, llm_input, emb_input)
    out_generic = run_case(his_id_generic, llm_input, emb_input)
    out_empty = run_case(his_id_empty, llm_input, emb_input)

    # Optional gate activation
    gate_val_mean = None
    if USE_GATE and hasattr(model, "gate"):
        try:
            dummy_task = profile_real.unsqueeze(1)
            dummy_user = profile_empty.unsqueeze(1)
            with torch.no_grad():
                gate_val_mean = torch.sigmoid(model.gate(torch.cat([dummy_task, dummy_user], dim=-1))).mean().item()
        except Exception:
            pass

    # Print
    print(f"\n=== {case_type} Query ===")
    print(f"UserIdx: {prof_idx} ({real_user_id}) | HistLen: {len(user_ctx['his_id'])}")
    print(persona_stats(profile_real, profile_generic, "generic"))
    print(persona_stats(profile_real, profile_empty, "empty"))
    if gate_val_mean is not None:
        print(f"    Avg Gate Act (real vs empty): {gate_val_mean:.4f}")
    print(f"Query: {query}")
    print(f"Real Persona → {out_real}")
    print(f"Random Generic Persona → {out_generic}")
    print(f"Empty Persona → {out_empty}")

print("\n=======================\n")
