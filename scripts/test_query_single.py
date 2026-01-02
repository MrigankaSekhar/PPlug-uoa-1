
import torch
import json
import random
import sys
import os
import numpy as np
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
CHEKPOINT_NUM=3435
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
    """Run model's own task→profile embedding path.
       Returns:
         - Neutral placeholder vector for empty persona (aligned to LLM dim)
         - Learned vectors from memmap for non-empty persona
    """
    with torch.no_grad():
        task_embs = model.obtain_task_emb(
            emb_input_ids=emb_input["input_ids"],
            emb_attention_mask=emb_input["attention_mask"],
            emb_token_type_ids=torch.zeros_like(emb_input["input_ids"])
        )

        # Empty history case
        if his_id.eq(0).all():
            memmap_src = model.his_train_memmap if model.training else model.his_dev_memmap
            neutral_vec_np = np.array(memmap_src[1:]).mean(axis=0, dtype=np.float32)  # (768,)
            neutral_vec = torch.from_numpy(neutral_vec_np).to(his_id.device).float().unsqueeze(0)  # (1,768)
            
            # Pass through align MLP if enabled
            if hasattr(model, "align_mlp") and model.use_align_mlp:
                neutral_vec = model.align_mlp(neutral_vec)

            return neutral_vec  # now shape matches real_emb (e.g., (1,512))

        # Non-empty case
        return model.obtain_profile_emb(his_id, task_embs)

def run_case(his_id, llm_input, emb_input):
    """Run inference + return decoded output.
       Fallback: if decoded text is blank, run without persona influence or return '[NO_OUTPUT]'.
    """
    with torch.no_grad():
        _, seqs = model.forward(
            llm_input_ids=llm_input["input_ids"],
            llm_attention_mask=llm_input["attention_mask"],
            labels=torch.zeros_like(llm_input["input_ids"]), # dummy labels
            emb_input_ids=emb_input["input_ids"],
            emb_attention_mask=emb_input["attention_mask"],
            emb_token_type_ids=torch.zeros_like(emb_input["input_ids"]),
            his_id=his_id
        )
    pred_text = llm_tokenizer.decode(seqs[0], skip_special_tokens=True).strip()

    # 🛠 Fallback for blank predictions
    if not pred_text or pred_text.strip(".") == "":
        print("[Fallback] Blank output detected — running without persona influence.")
        his_id_zero = torch.zeros_like(his_id)  # empty persona
        with torch.no_grad():
            _, seqs2 = model.forward(
                llm_input_ids=llm_input["input_ids"],
                llm_attention_mask=llm_input["attention_mask"],
                labels=None,
                emb_input_ids=emb_input["input_ids"],
                emb_attention_mask=emb_input["attention_mask"],
                emb_token_type_ids=torch.zeros_like(emb_input["input_ids"]),
                his_id=his_id_zero
            )
        pred_text = llm_tokenizer.decode(seqs2[0], skip_special_tokens=True).strip()

        # Final safety — if still blank or dots, insert a placeholder
        if not pred_text or pred_text.strip(".") == "":
            pred_text = "[NO_OUTPUT]"

    return pred_text

def persona_stats(real_emb, other_emb, other_label):
    cos = F.cosine_similarity(real_emb, other_emb).item()
    return f"{other_label}: cos={cos:.4f}, norm_real={real_emb.norm().item():.2f}, norm_{other_label}={other_emb.norm().item():.2f}"

# --- Query sets ---
unseen_queries = [
    ("UNSEEN", "What is the score of the following review on a scale of 1 to 5? just answer with 1, 2, 3, 4, or 5 without further explanation. review: The headphones arrived quickly and were packaged well. Sound quality is good for the price, although the ear pads feel a bit cheap。",
     random.randrange(len(profiles))),
    ("UNSEEN", "What is the score of the following review on a scale of 1 to 5? just answer with 1, 2, 3, 4, or 5 without further explanation. review: The book was fast-paced and entertaining, but some plot points were predictable. Still enjoyed it overall。",
     random.randrange(len(profiles))),
    # Predicting numeric rating from review + category preference
    ("UNSEEN-GRAPH", 
     "User previously favored 'Strategy Board Games'. Predict the star rating (1-5) they would give to this review: 'Gameplay is deep but turns are slow. Artwork is beautiful though.'",
     random.randrange(len(profiles))),

    # Binary preference for category
    ("UNSEEN-GRAPH", 
     "Based on their profile history, which do they prefer: 'Jazz Music Albums' or 'Pop Music Albums'? Review: 'The smooth saxophone melodies were relaxing and the live recording felt intimate.'",
     random.randrange(len(profiles))),

    # Item prediction from connected categories
    ("UNSEEN-GRAPH",
     "User often buys items in 'Outdoor Camping Gear'. Would they purchase this item? review: 'Compact, lightweight tent with waterproof material. Easy to pack and set up.' Answer: yes or no.",
     random.randrange(len(profiles))),

    # Graph-based rating with clear instruction
    ("UNSEEN-GRAPH",
     "Given the user's history in 'Science Fiction Movies', how likely (1-5) to rate this film positively? Review: 'Futuristic cityscapes were stunning but plot lacked depth.'",
     random.randrange(len(profiles)))
]

seen_queries = [("SEEN", q["input"], i) for i, q in enumerate(dev_questions[:2])]
test_queries = unseen_queries + seen_queries

# --- Smart Graph-based Query Generation ---
graph_dir = "../graph_emb"
his_to_graph_path = os.path.join(graph_dir, "task_3_his_to_graph.json")

if os.path.exists(his_to_graph_path):
    his_to_graph = json.load(open(his_to_graph_path))

    try:
        # Import node maps from the graph building module
        from extention import compute_graph_emb_generic_npy as graph_mod
        node_maps = graph_mod.node_maps  # dict per type: "Category", "Review", "Item", etc.
    except Exception as e:
        print(f"⚠️ Could not load node_maps from graph module: {e}")
        node_maps = {}

    # Build reverse lookup for category nodes: id -> category name
    id_to_category = {v: k for k, v in node_maps.get("Category", {}).items()}

    # Map each user index to a set of category names from their history
    category_nodes = {}
    for user_idx, profile in enumerate(profiles):
        cats = set()
        for hid in profile["his_id"]:
            node_id = his_to_graph.get(str(hid))
            if node_id is not None and node_id in id_to_category:
                cats.add(id_to_category[node_id])
        category_nodes[user_idx] = cats

    # Generate personalized UNSEEN-GRAPH queries for each user
    smart_graph_queries = []
    all_categories = list(id_to_category.values())
    for u_idx, cats in category_nodes.items():
        if not cats:
            continue

        target_cat = random.choice(list(cats))
        # pick a category they have NOT interacted with — if possible
        non_target_choices = [c for c in all_categories if c not in cats]
        non_target_cat = random.choice(non_target_choices) if non_target_choices else "OtherCategory"

        # Preference query
        smart_graph_queries.append((
            "UNSEEN-GRAPH",
            f"Based on their profile history, which do they prefer: '{target_cat}' or '{non_target_cat}'? Review: This item belongs to {target_cat}.",
            u_idx
        ))

        # Rating forecast query
        smart_graph_queries.append((
            "UNSEEN-GRAPH",
            f"Given the user's history in '{target_cat}', how likely (1-5) to rate this item positively? Review: High quality and meets expectations.",
            u_idx
        ))

        # Purchase likelihood query
        smart_graph_queries.append((
            "UNSEEN-GRAPH",
            f"User has history in '{target_cat}'. Would they purchase this item? Answer: yes or no. Review: Durable and matches previous purchases.",
            u_idx
        ))

    unseen_queries += smart_graph_queries
else:
    print("⚠️ his_to_graph.json not found — skipping smart graph query generation.")

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
                gate_val_mean = torch.sigmoid(
                    model.gate(torch.cat([dummy_task, dummy_user], dim=-1))
                ).mean().item()
        except Exception:
            pass

    # Print basic stats
    print(f"\n=== {case_type} Query ===")
    print(f"UserIdx: {prof_idx} ({real_user_id}) | HistLen: {len(user_ctx['his_id'])}")
    print(persona_stats(profile_real, profile_generic, "generic"))
    print(persona_stats(profile_real, profile_empty, "empty"))
    if gate_val_mean is not None:
        print(f"    Avg Gate Act (real vs empty): {gate_val_mean:.4f}")

    # 🔍 Always log embedding diagnostics now (not just for UNSEEN-GRAPH)
    cos_rg = F.cosine_similarity(profile_real, profile_generic).item()
    l2_rg = torch.norm(profile_real - profile_generic).item()
    print(f"    [EmbDiag] Real vs Generic → cos={cos_rg:.4f}, L2 diff={l2_rg:.4f}")
    cos_re = F.cosine_similarity(profile_real, profile_empty).item()
    l2_re = torch.norm(profile_real - profile_empty).item()
    print(f"    [EmbDiag] Real vs Empty   → cos={cos_re:.4f}, L2 diff={l2_re:.4f}")

    # Output predictions
    print(f"Query: {query}")
    print(f"Real Persona → {out_real}")
    print(f"Random Generic Persona → {out_generic}")
    print(f"Empty Persona → {out_empty}")

print("\n=======================\n")
