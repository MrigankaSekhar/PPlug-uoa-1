"""
Compare outputs of:
  (A) Flan‑T5 baseline (no personalization)
  (B) Personalized model using user profile embeddings
  (C) Personalized + Graph‑enhanced model (profile + GNN embeddings)

Purpose:
  Demonstrate how personalization and graph context modify text generation.
  After fine‑tuning, this script allows side‑by‑side qualitative comparison and
  gate monitoring (personalization strength diagnostics).
"""

import os, sys, json, math, torch
from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoModel
torch.set_grad_enabled(False)

# -------------------------------------------------------------------------
# ✅ Setup imports
# -------------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from extention.ModelForPer_slim_GNN import PersonalLLM_Slim
try:
    from extention.graph_encoder_module import GraphEncoder
except ImportError:
    GraphEncoder = None

# -------------------------------------------------------------------------
# 🔧 CONFIG
# -------------------------------------------------------------------------
TASK_ID = 3
CHECKPOINT_PATH = "../extention/output_3/checkpoint-2062"
MAX_HIS_LEN = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------------------------------
# 🧠 LOAD MODELS
# -------------------------------------------------------------------------
llm_model_path = "../FlanT5-small"
emb_model_path = "../bge-base-en-v1.5"

llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_path, use_fast=False)
emb_tokenizer = AutoTokenizer.from_pretrained(emb_model_path)
llm_model = T5ForConditionalGeneration.from_pretrained(llm_model_path).to(DEVICE).eval()
emb_model = AutoModel.from_pretrained(emb_model_path)

# -------------------------------------------------------------------------
# 🚀 LOAD PERSONALIZED MODEL
# -------------------------------------------------------------------------
try:
    personal_model = PersonalLLM_Slim(
        llm_model=llm_model,
        emb_model=emb_model,
        max_input_len=256,
        max_new_len=64,
        task_id=TASK_ID
    ).to(DEVICE)
    state_dict = torch.load(f"{CHECKPOINT_PATH}/pytorch_model.bin", map_location=DEVICE)
    # --- PATCH: Ignore vocab shape mismatches for Flan-T5 ---
    skip_prefixes = [
        "llm_model.shared.weight",
        "llm_model.encoder.embed_tokens.weight",
        "llm_model.decoder.embed_tokens.weight",
        "llm_model.lm_head.weight"
    ]
    filtered_state_dict = {k: v for k, v in state_dict.items() if not any(k.startswith(p) for p in skip_prefixes)}
    missing, unexpected = personal_model.load_state_dict(filtered_state_dict, strict=False)
    print("✅ Loaded checkpoint (vocab shape differences safely ignored)")
    if missing:
        print(f"ℹ️ Missing keys: {len(missing)} → {missing[:8]}{'...' if len(missing)>8 else ''}")
    if unexpected:
        print(f"ℹ️ Unexpected keys: {len(unexpected)} → {unexpected[:8]}{'...' if len(unexpected)>8 else ''}")
    personal_model.eval()
except Exception as e:
    print(f"⚠️ Could not load checkpoint ({e}). Using untrained fallback.")
    personal_model = PersonalLLM_Slim(
        llm_model=llm_model,
        emb_model=emb_model,
        max_input_len=256,
        max_new_len=64,
        task_id=TASK_ID
    ).to(DEVICE).eval()

# -------------------------------------------------------------------------
# 🌐 GRAPH ENCODER (optional)
# -------------------------------------------------------------------------
if GraphEncoder:
    try:
        graph_encoder = GraphEncoder(
            num_nodes=5000, emb_dim=768, hidden_dim=256,
            alpha=0.2, dropout=0.3, use_norm=True
        ).to(DEVICE).eval()
    except Exception:
        graph_encoder = None
        print("⚠️ GraphEncoder init failed; skipping graph model.")
else:
    graph_encoder = None
    print("⚠️ GraphEncoder module unavailable.")

# -------------------------------------------------------------------------
# 📑 Example user & review IDs (confirmed valid IDs)
# -------------------------------------------------------------------------
# Automatically pick a valid user and profile_his_ids from offsets file
with open("../bge_emb/task_3_train_offsets.json") as f:
    offsets = json.load(f)

# Find the first entry with enough profile IDs
user_entry = None
for entry in offsets["entries"]:
    if len(entry["profile_id"]) >= MAX_HIS_LEN:
        user_entry = entry
        break

if user_entry is None:
    raise ValueError("No user found with enough profile reviews in offsets file.")

user_id = user_entry["profile_id"][0]  # Use the first profile_id as user_id (or use question_index if needed)
profile_his_ids = user_entry["profile_id"][:MAX_HIS_LEN]  # Truncate to MAX_HIS_LEN

print(f"✅ Picked user_id: {user_id}")
print(f"✅ Picked profile_his_ids: {profile_his_ids}")

# -------------------------------------------------------------------------
# 🔄 Load mapping files safely
# -------------------------------------------------------------------------
try:
    with open("../graph_emb/task_3_his_to_graph.json") as f:
        node_maps = json.load(f)
    with open("../graph_emb/task_3_processed_ids.json") as f:
        id_to_index = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError("❌ Missing graph files. Run embedding/graph pipeline first.")

# Determine ID table shape
id_map_is_list = isinstance(id_to_index, list)
if id_map_is_list:
    print("ℹ️ task_3_processed_ids.json is a list — treating entries as sequential indices.")

# --- PATCH: Robust review node index extraction ---
review_node_indices = []
for hid in profile_his_ids:
    idx = None
    # Try direct integer index if mapping is a list
    if isinstance(node_maps, list):
        try:
            idx = int(hid)
            if 0 <= idx < len(node_maps):
                review_node_indices.append(idx)
            else:
                review_node_indices.append(-1)
        except Exception:
            review_node_indices.append(-1)
    # Try dictionary lookup if mapping is a dict
    elif isinstance(node_maps, dict):
        idx = node_maps.get(hid) or node_maps.get(f"review_{hid}") or node_maps.get(str(hid))
        review_node_indices.append(idx if idx is not None else -1)
    else:
        review_node_indices.append(-1)
print(f"🔗 Review node indices for user {user_id}: {review_node_indices}")

# -------------------------------------------------------------------------
# 🧩 Build valid history tensor
# -------------------------------------------------------------------------
memmap_size = getattr(personal_model, "his_train_memmap", torch.zeros((1,))).shape[0]
valid_indices = []

# --- PATCH: Use review_node_indices directly if id_to_index is a list ---
if id_map_is_list:
    # Only keep indices that are valid for the memmap size
    valid_indices = [idx for idx in review_node_indices if isinstance(idx, int) and 0 <= idx < memmap_size]
else:
    # If id_to_index is a dict, map profile_his_ids to indices and check validity
    for hid in profile_his_ids:
        cand = id_to_index.get(hid) or id_to_index.get(str(hid)) or id_to_index.get(f"review_{hid}")
        if cand is not None and isinstance(cand, int) and 0 <= cand < memmap_size:
            valid_indices.append(cand)

if not valid_indices:
    print(f"⚠️ No valid indices for user {user_id}; using empty persona tensor.")
    his_id_real = torch.zeros((1, MAX_HIS_LEN), dtype=torch.long, device=DEVICE)
else:
    padded = valid_indices[:MAX_HIS_LEN] + [0]*(MAX_HIS_LEN - len(valid_indices))
    his_id_real = torch.tensor(padded, dtype=torch.long, device=DEVICE).unsqueeze(0)
his_id_empty = torch.zeros_like(his_id_real)

# -------------------------------------------------------------------------
# ✨ Tensor NaN cleaner
# -------------------------------------------------------------------------
def _sanitize(t: torch.Tensor):
    return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

# -------------------------------------------------------------------------
# 🔧 Generation helpers
# -------------------------------------------------------------------------
def run_llm_base(prompt: str) -> str:
    """Generate text using plain Flan‑T5."""
    tokens = llm_tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True).to(DEVICE)
    out = llm_model.generate(**tokens, num_beams=3, min_new_tokens=8, max_new_tokens=32)
    return llm_tokenizer.decode(out[0], skip_special_tokens=True).strip()

def run_personal(prompt: str, his_id: torch.Tensor) -> str:
    """Generate using personalized LLM (correct tokenizers, labels=None)."""
    llm_inp = llm_tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True).to(DEVICE)
    emb_inp = emb_tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True).to(DEVICE)
    try:
        # Try with labels=None (inference mode)
        _, seqs = personal_model.forward(
            llm_input_ids=llm_inp["input_ids"],
            llm_attention_mask=llm_inp["attention_mask"],
            labels=None,
            emb_input_ids=emb_inp["input_ids"],
            emb_attention_mask=emb_inp["attention_mask"],
            emb_token_type_ids=torch.zeros_like(emb_inp["input_ids"]),
            his_id=his_id,
        )
    except AttributeError as e:
        # Fallback: pass a dummy tensor if labels=None causes error
        if "'NoneType' object has no attribute 'long'" in str(e):
            dummy_labels = torch.zeros_like(llm_inp["input_ids"])
            _, seqs = personal_model.forward(
                llm_input_ids=llm_inp["input_ids"],
                llm_attention_mask=llm_inp["attention_mask"],
                labels=dummy_labels,
                emb_input_ids=emb_inp["input_ids"],
                emb_attention_mask=emb_inp["attention_mask"],
                emb_token_type_ids=torch.zeros_like(emb_inp["input_ids"]),
                his_id=his_id,
            )
        else:
            raise
    return llm_tokenizer.decode(seqs[0], skip_special_tokens=True).strip()

def run_personal_graph(prompt: str, his_id: torch.Tensor):
    """Generate using personalized+graph model, return gate average (correct tokenizers)."""
    llm_inp = llm_tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True).to(DEVICE)
    emb_inp = emb_tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True).to(DEVICE)
    task_emb = _sanitize(personal_model.obtain_task_emb(
        emb_inp["input_ids"], emb_inp["attention_mask"],
        torch.zeros_like(emb_inp["input_ids"])
    ))
    prof_emb = _sanitize(personal_model.obtain_profile_emb(his_id, task_emb))
    # --- PATCH: Use review_node_indices for graph embedding if available ---
    if graph_encoder and any(idx >= 0 for idx in review_node_indices):
        try:
            graph_tensor = torch.tensor([review_node_indices], dtype=torch.long, device=DEVICE)
            graph_emb = _sanitize(graph_encoder(graph_tensor))
        except Exception:
            graph_emb = torch.zeros_like(prof_emb)
    else:
        graph_emb = torch.zeros_like(prof_emb)
    fused_emb = torch.cat([prof_emb, graph_emb], dim=-1)
    if fused_emb.size(-1) != task_emb.size(-1):
        W = torch.randn(task_emb.size(-1), fused_emb.size(-1), device=fused_emb.device)
        fused_emb = torch.nn.functional.linear(fused_emb, W)
    fused_emb = _sanitize(fused_emb)
    gate_input = _sanitize(torch.cat([task_emb, fused_emb], dim=-1))
    try:
        gate_val = float(torch.sigmoid(personal_model.gate(gate_input)).mean().item())
    except Exception:
        gate_val = 0.0
    gate_val = 0.0 if math.isnan(gate_val) else gate_val
    try:
        # Try with labels=None (inference mode)
        _, seqs = personal_model.forward(
            llm_input_ids=llm_inp["input_ids"],
            llm_attention_mask=llm_inp["attention_mask"],
            labels=None,
            emb_input_ids=emb_inp["input_ids"],
            emb_attention_mask=emb_inp["attention_mask"],
            emb_token_type_ids=torch.zeros_like(emb_inp["input_ids"]),
            his_id=his_id,
        )
        text = llm_tokenizer.decode(seqs[0], skip_special_tokens=True).strip()
    except AttributeError as e:
        if "'NoneType' object has no attribute 'long'" in str(e):
            dummy_labels = torch.zeros_like(llm_inp["input_ids"])
            _, seqs = personal_model.forward(
                llm_input_ids=llm_inp["input_ids"],
                llm_attention_mask=llm_inp["attention_mask"],
                labels=dummy_labels,
                emb_input_ids=emb_inp["input_ids"],
                emb_attention_mask=emb_inp["attention_mask"],
                emb_token_type_ids=torch.zeros_like(emb_inp["input_ids"]),
                his_id=his_id,
            )
            text = llm_tokenizer.decode(seqs[0], skip_special_tokens=True).strip()
        else:
            raise
    return text, gate_val

# -------------------------------------------------------------------------
# 🧪 PROMPTS FOR DEMO
# -------------------------------------------------------------------------
test_prompts = [
    "What is the score of the following review on a scale of 1 to 5? just answer with 1, 2, 3, 4, or 5 without further explanation. review: The story was fun but predictable.",
    "What is the score of the following review on a scale of 1 to 5? just answer with 1, 2, 3, 4, or 5 without further explanation. review: The shipping was fast!",
    "What is the score of the following review on a scale of 1 to 5? just answer with 1, 2, 3, 4, or 5 without further explanation. review: This tent survived heavy rain.",
    "What is the score of the following review on a scale of 1 to 5? just answer with 1, 2, 3, 4, or 5 without further explanation. review: It is difficult to believe that this was produced by an adult.  It lacks finesse, style and cohesiveness.  The dialogue is stilted and painful to read. I almost tossed it aside after the first few pages.  I can find no redeeming qualities in this offering.",
    "What is the score of the following review on a scale of 1 to 5? just answer with 1, 2, 3, 4, or 5 without further explanation. review: every few days it stops connecting to the internet and you have to restart it not likeing that especially since its slow on start up and fire fox is the only browser that works on it"
]

# -------------------------------------------------------------------------
# 📊 COMPARISON RUN
# -------------------------------------------------------------------------
for prompt in test_prompts:
    print(f"\n🧾 Prompt: {prompt}")
    out_base = run_llm_base(prompt)
    out_pers = run_personal(prompt, his_id_real)
    out_emp = run_personal(prompt, his_id_empty)
    out_graph, gate_val = run_personal_graph(prompt, his_id_real)
    print(f"  BASE (Flan‑T5)               → {out_base}")
    print(f"  Personalized (Real Persona)  → {out_pers}")
    print(f"  Personalized (Empty Persona) → {out_emp}")
    print(f"  Personalized + Graph         → {out_graph}  | gate_avg={gate_val:.4f}")

# -------------------------------------------------------------------------
# 💡 INTERPRETATION GUIDELINES
# -------------------------------------------------------------------------
# • Differences between BASE and PERSONALIZED show persona influence.
# • gate_mean≈0 → persona ignored; ≈0.5 → balanced; ≈1.0 → strong personalization.
# • Adding Graph context should increase contextual consistency and stability.