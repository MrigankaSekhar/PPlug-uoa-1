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
import random
import re
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
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
CHECKPOINT_PATH = "../extention/output_3/checkpoint-129"
MAX_HIS_LEN = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------------------------------
# 🔧 DEMO MODES (what to print)
# -------------------------------------------------------------------------
SHOW_BASE = True
SHOW_PERSONA_REAL = True
SHOW_PERSONA_EMPTY = False
SHOW_GRAPH = False   


# -------------------------------------------------------------------------
# 🔧 DEV GOLD EVAL (optional)
# -------------------------------------------------------------------------
USE_DEV_GOLD = True
DEV_SUBSET = True
DEV_SAMPLE_SIZE = 8

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
personal_model = PersonalLLM_Slim(
    llm_model=llm_model,
    emb_model=emb_model,
    max_input_len=256,
    max_new_len=64,
    task_id=TASK_ID
).to(DEVICE)
personal_model.llm_tokenizer = llm_tokenizer

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

def extract_numeric_rating(text: str):
    m = re.search(r"\b([1-5])(\.0+)?\b", str(text))
    return int(m.group(1)) if m else None

def _seqs_to_text(seqs) -> str:
    """
    PersonalLLM_Slim.forward() may return:
      - token id tensors (typical HF generate), or
      - already-decoded strings / list[str] (custom model code).
    This helper normalizes to a string safely.
    """
    if seqs is None:
        return ""

    first = seqs[0] if isinstance(seqs, (list, tuple)) else seqs

    # Case 1: already a python string
    if isinstance(first, str):
        return first.strip()

    # Case 2: list of ints
    if isinstance(first, list) and (len(first) == 0 or isinstance(first[0], int)):
        return llm_tokenizer.decode(first, skip_special_tokens=True).strip()

    # Case 3: torch tensor / numpy array of ids
    if torch.is_tensor(first):
        ids = first.detach().cpu().tolist()
        return llm_tokenizer.decode(ids, skip_special_tokens=True).strip()

    # Fallback: stringify
    return str(first).strip()

def _forward_out_to_text(out) -> str:
    """
    Handles various PersonalLLM_Slim.forward() return formats:
      - (loss, seqs)
      - (loss, logits)
      - {'loss':..., 'logits':...}
    If logits are returned, decodes only the FIRST generated token (rating).
    """
    if out is None:
        return ""

    def _decode_first_from_logits(logits: torch.Tensor) -> str:
        # logits: (B, T, V) -> take first position only
        first_token_id = logits[:, 0, :].argmax(dim=-1)  # (B,)
        return llm_tokenizer.decode(first_token_id[0].detach().cpu().tolist(), skip_special_tokens=True).strip()

    # dict output
    if isinstance(out, dict):
        if "sequences" in out:
            return _seqs_to_text(out["sequences"])
        if "logits" in out and torch.is_tensor(out["logits"]) and out["logits"].dim() == 3:
            return _decode_first_from_logits(out["logits"])
        return str(out).strip()

    # tuple/list output
    if isinstance(out, (list, tuple)) and len(out) >= 2:
        second = out[1]
        if torch.is_tensor(second) and second.dim() == 3:
            return _decode_first_from_logits(second)
        return _seqs_to_text(second)

    return str(out).strip()

# -------------------------------------------------------------------------
# 🔧 Generation helpers
# -------------------------------------------------------------------------
def run_llm_base(prompt: str) -> str:
    """Generate text using plain Flan‑T5."""
    tokens = llm_tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True).to(DEVICE)
    out = llm_model.generate(**tokens, num_beams=3, min_new_tokens=8, max_new_tokens=32)
    return llm_tokenizer.decode(out[0], skip_special_tokens=True).strip()

def run_personal(prompt: str, his_id: torch.Tensor) -> str:
    llm_inp = llm_tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True).to(DEVICE)
    emb_inp = emb_tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True).to(DEVICE)

    dummy_labels = torch.zeros_like(llm_inp["input_ids"])

    out = personal_model.forward(
        llm_input_ids=llm_inp["input_ids"],
        llm_attention_mask=llm_inp["attention_mask"],
        labels=dummy_labels,
        emb_input_ids=emb_inp["input_ids"],
        emb_attention_mask=emb_inp["attention_mask"],
        emb_token_type_ids=torch.zeros_like(emb_inp["input_ids"]),
        his_id=his_id,
    )
    return _forward_out_to_text(out)

def run_personal_graph(prompt: str, his_id: torch.Tensor, graph_node_ids: torch.Tensor = None, session_ids: torch.Tensor = None):
    llm_inp = llm_tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True).to(DEVICE)
    emb_inp = emb_tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True).to(DEVICE)

    dummy_labels = torch.zeros_like(llm_inp["input_ids"])

    forward_kwargs = dict(
        llm_input_ids=llm_inp["input_ids"],
        llm_attention_mask=llm_inp["attention_mask"],
        labels=dummy_labels,
        emb_input_ids=emb_inp["input_ids"],
        emb_attention_mask=emb_inp["attention_mask"],
        emb_token_type_ids=torch.zeros_like(emb_inp["input_ids"]),
        his_id=his_id,
    )

    if graph_node_ids is not None:
        forward_kwargs["graph_node_ids"] = graph_node_ids.to(DEVICE)

    if session_ids is not None:
        forward_kwargs["session_ids"] = session_ids.to(DEVICE)

    _, seqs = personal_model.forward(**forward_kwargs)
    return _seqs_to_text(seqs)


# -------------------------------------------------------------------------
# 🧪 PROMPTS FOR DEMO / OR DEV SET SAMPLING
# -------------------------------------------------------------------------
if USE_DEV_GOLD:
    if DEV_SUBSET:
        questions_path = "../LaMP_time_3_subset/dev_questions.json"
        outputs_path   = "../LaMP_time_3_subset/dev_outputs.json"
    else:
        questions_path = "../LaMP_time_3/dev_questions.json"
        outputs_path   = "../LaMP_time_3/dev_outputs.json"

    if not (os.path.exists(questions_path) and os.path.exists(outputs_path)):
        raise FileNotFoundError(f"Missing dev files: {questions_path} or {outputs_path}")

    with open(questions_path, "r") as fq:
        questions_data = json.load(fq)
    with open(outputs_path, "r") as fo:
        outputs_data = json.load(fo)

    id_to_gold = {g["id"]: str(g["output"]).strip() for g in outputs_data.get("golds", [])}

    # sample entries that have gold
    candidates = [q for q in questions_data if q.get("id") in id_to_gold]
    if not candidates:
        raise ValueError("No dev questions matched gold IDs.")

    sampled = random.sample(candidates, k=min(DEV_SAMPLE_SIZE, len(candidates)))

    # Each item: {"id","prompt","gold"}
    test_items = []
    for q in sampled:
        qid = q.get("id")
        prompt = q.get("input", "")
        gold = id_to_gold.get(qid, "")
        test_items.append({"id": qid, "prompt": prompt, "gold": gold})
else:
    test_items = [{"id": None, "prompt": p, "gold": None} for p in [
        "What is the score of the following review on a scale of 1 to 5? just answer with 1, 2, 3, 4, or 5 without further explanation. review: The story was fun but predictable.",
        "What is the score of the following review on a scale of 1 to 5? just answer with 1, 2, 3, 4, or 5 without further explanation. review: The shipping was fast!",
        "What is the score of the following review on a scale of 1 to 5? just answer with 1, 2, 3, 4, or 5 without further explanation. review: This tent survived heavy rain.",
    ]]

# -------------------------------------------------------------------------
# 📊 COMPARISON RUN (+ GOLD)
# -------------------------------------------------------------------------
rows = []

for item in test_items:
    prompt = item["prompt"]
    gold = item["gold"]
    qid = item["id"]

    print(f"\n🧾 ID: {qid} | Prompt: {prompt}")
    if gold is not None:
        print(f"  GOLD                        → {gold}")

    out_base = out_pers = out_emp = out_graph = None

    if SHOW_BASE:
        out_base = run_llm_base(prompt)
        print(f"  BASE (Flan‑T5)               → {out_base}")

    if SHOW_PERSONA_REAL:
        out_pers = run_personal(prompt, his_id_real)
        print(f"  Personalized (Real Persona)  → {out_pers}")

    if SHOW_PERSONA_EMPTY:
        out_emp = run_personal(prompt, his_id_empty)
        print(f"  Personalized (Empty Persona) → {out_emp}")

    if SHOW_GRAPH:
        graph_ids_tensor = torch.tensor([review_node_indices], dtype=torch.long, device=DEVICE)
        out_graph = run_personal_graph(prompt, his_id_real, graph_node_ids=graph_ids_tensor)
        print(f"  Personalized + Graph         → {out_graph}")

    rows.append({
        "id": qid,
        "gold": gold,
        "base": out_base,
        "pers": out_pers,
        "empty": out_emp,
        "graph": out_graph
    })

# -------------------------------------------------------------------------
# 📈 METRICS (numeric match + RMSE/MAE) when gold exists
# -------------------------------------------------------------------------
if USE_DEV_GOLD:
    def _metric_block(name, preds):
        gold_nums = []
        pred_nums = []
        for r in rows:
            if r["gold"] is None:
                continue
            g = extract_numeric_rating(r["gold"])
            p = extract_numeric_rating(preds(r))
            if g is not None and p is not None:
                gold_nums.append(g)
                pred_nums.append(p)

        if not gold_nums:
            print(f"\n=== METRICS: {name} ===")
            print("  (no parsable numeric pairs)")
            return

        acc = float(np.mean([int(g == p) for g, p in zip(gold_nums, pred_nums)]))
        rmse = float(mean_squared_error(gold_nums, pred_nums, squared=False))
        mae = float(mean_absolute_error(gold_nums, pred_nums))
        print(f"\n=== METRICS: {name} ===")
        print(f"  n={len(gold_nums)} | acc={acc:.3f} | rmse={rmse:.3f} | mae={mae:.3f}")

    if SHOW_BASE:
        _metric_block("BASE", lambda r: r["base"])
    if SHOW_PERSONA_REAL:
        _metric_block("PERSONAL (REAL)", lambda r: r["pers"])
    if SHOW_PERSONA_EMPTY:
        _metric_block("PERSONAL (EMPTY)", lambda r: r["empty"])
    if SHOW_GRAPH:
        _metric_block("PERSONAL + GRAPH", lambda r: r["graph"])

# -------------------------------------------------------------------------
# 💡 INTERPRETATION GUIDELINES
# -------------------------------------------------------------------------
# • Differences between BASE and PERSONALIZED show persona influence.
# • gate_mean≈0 → persona ignored; ≈0.5 → balanced; ≈1.0 → strong personalization.
# • Adding Graph context should increase contextual consistency and stability.