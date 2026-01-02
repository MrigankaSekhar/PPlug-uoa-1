
import torch
import json
from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoModel
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix, cohen_kappa_score, matthews_corrcoef
import numpy as np
import sys
import os
import math
import networkx as nx  # for graph degree analysis
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from extention.ModelForPer_slim_GNN import PersonalLLM_Slim
import datetime

# ======== summary ========
# Tests a fine-tuned PersonalLLM_Slim model on LaMP dev set.
# Computes Accuracy, Macro F1, classification report & confusion matrix.
# Writes all results to a timestamped file in output_3/metrics.
# =======================

USE_PROFILE = True   # personalization ON/OFF
USE_GATE = True      # gating ON/OFF
TASK_ID = 3          # relevant LaMP task ID
CHEKPOINT_NUM=3435
# Prepare output directory and timestamped file path
metrics_dir = "./output_3/metrics"
os.makedirs(metrics_dir, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FILE = os.path.join(metrics_dir, f"{timestamp}_metrics.txt")

# Helper: append text to file
def write_to_file(text):
    with open(OUTPUT_FILE, "a") as f:
        f.write(text + "\n")

# 1️⃣ Load tokenizer + base backbone from FlanT5-small
llm_tokenizer = AutoTokenizer.from_pretrained("../FlanT5-small", use_fast=False)
emb_tokenizer = AutoTokenizer.from_pretrained("../bge-base-en-v1.5")

llm_model = T5ForConditionalGeneration.from_pretrained("../FlanT5-small")

# 2️⃣ Load fine-tuned checkpoint weights
ckpt_path = f"../extention/output_3/checkpoint-{CHEKPOINT_NUM}/pytorch_model.bin"
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"❌ Fine-tuned checkpoint not found at {ckpt_path}")
state_dict = torch.load(ckpt_path, map_location="cpu")
llm_model.load_state_dict(state_dict, strict=False)

# 3️⃣ Load embedding model
emb_model = AutoModel.from_pretrained("../bge-base-en-v1.5")

# 4️⃣ Initialize model
model = PersonalLLM_Slim(
    llm_model=llm_model,
    emb_model=emb_model,
    max_input_len=256,
    max_new_len=32,
    task_id=TASK_ID,
    # cfg={"use_profile_emb": USE_PROFILE, "use_gate": USE_GATE}
)
model.eval()

# 5️⃣ Load dev set
with open("../LaMP_time_3_subset_id/dev_profile.json") as f:
    profiles = [json.loads(l) for l in f]

preds_str, refs_str = [], []

# 6️⃣ Run inference
for sample in profiles:
    if "input_text" in sample:
        user_query = sample["input_text"]
    elif "input" in sample:
        user_query = sample["input"]
    else:
        raise KeyError(f"Missing 'input_text' or 'input'. Keys: {list(sample.keys())}")

    ref = str(sample.get("output", sample.get("label", ""))).strip()
    his_id = torch.tensor(sample["his_id"]).unsqueeze(0)

    llm_input = llm_tokenizer(user_query, return_tensors="pt", max_length=256, truncation=True)
    emb_input = emb_tokenizer(user_query, return_tensors="pt", max_length=256, truncation=True)

    with torch.no_grad():
        _, sequences = model.forward(
            llm_input_ids=llm_input["input_ids"],
            llm_attention_mask=llm_input["attention_mask"],
            labels=torch.zeros_like(llm_input["input_ids"]), # dummy labels
            emb_input_ids=emb_input["input_ids"],
            emb_attention_mask=emb_input["attention_mask"],
            emb_token_type_ids=torch.zeros_like(emb_input["input_ids"]),
            his_id=his_id
        )
    pred_text = llm_tokenizer.decode(sequences[0], skip_special_tokens=True).strip()

    preds_str.append(pred_text)
    refs_str.append(ref)

# 7️⃣ Robust numeric parsing for ordinal metrics
# ----------------------------------------------
# LaMP-3 ratings are *meant* to be numeric (1–5, sometimes decimal), but the
# model might output them in text form like "3", "3.5", "rating: 4", or "4 stars".
# This helper safely extracts the first number from any string.
import re

def safe_number_parse(x):
    """
    Try to parse a value into a float:
    - If already int/float → return as float
    - If string with a number inside → extract first number (int/decimal)
    - Else → return None (non-numeric)
    """
    # Already numeric (int or float type)
    if isinstance(x, (int, float)):
        return float(x)
    # Try direct float parsing (e.g., "4.5", "3")
    try:
        return float(x)
    except (ValueError, TypeError):
        # Try regex to find a number inside messy text
        match = re.search(r"[-+]?\d*\.?\d+", str(x))
        if match:
            try:
                return float(match.group())
            except ValueError:
                return None
        return None

# Apply parsing to predictions and references
numeric_values_preds = [safe_number_parse(p) for p in preds_str]
numeric_values_refs = [safe_number_parse(r) for r in refs_str]

# Flag: True if *all* parsed successfully → we can compute MAE/RMSE & Graph breakdown
if all(v is not None for v in numeric_values_preds + numeric_values_refs):
    preds_int = numeric_values_preds
    refs_int = numeric_values_refs
    numeric_for_ordinals = True
else:
    preds_int, refs_int = preds_str, refs_str
    numeric_for_ordinals = False

# 8️⃣ Classification metrics (work on numeric OR string labels)
# --------------------------------------------------------------
acc = accuracy_score(refs_int, preds_int)                     # Exact match rate
f1_macro = f1_score(refs_int, preds_int, average="macro")      # Equal weight to classes
f1_weighted = f1_score(refs_int, preds_int, average="weighted")# Weighted by class frequency
kappa = cohen_kappa_score(refs_int, preds_int)                 # Agreement beyond chance
mcc = matthews_corrcoef(refs_int, preds_int)                   # Balanced correlation score

# 9️⃣ Ordinal regression metrics — only for numeric
# -------------------------------------------------
if numeric_for_ordinals:
    mae = np.mean([abs(p - r) for p, r in zip(preds_int, refs_int)])   # Mean Absolute Error
    rmse = math.sqrt(np.mean([(p - r) ** 2 for p, r in zip(preds_int, refs_int)])) # Root Mean Square Error
else:
    mae, rmse = None, None

# 🔟 Summary block — metrics overview
summary = "\n=======================\n"
summary += f"📊 Accuracy: {acc:.4f}         # Exact match rate\n"
summary += f"📊 Macro F1: {f1_macro:.4f}   # Balanced per-class performance\n"
summary += f"📊 Weighted F1: {f1_weighted:.4f} # Adjusts for class imbalance\n"
summary += f"📊 Cohen's Kappa: {kappa:.4f} # Agreement beyond chance\n"
summary += f"📊 Matthews CC: {mcc:.4f}     # Correlation between pred/true labels\n"
if numeric_for_ordinals:
    summary += f"📊 MAE (ratings): {mae:.4f}   # Avg absolute rating error\n"
    summary += f"📊 RMSE (ratings): {rmse:.4f} # Penalizes large errors more\n"
else:
    summary += "📊 MAE / RMSE skipped — labels non-numeric\n"
summary += "=======================\n"

print(summary)
write_to_file(summary)

# 1️⃣1️⃣ Graph-aware breakdown — only if numeric_for_ordinals
# -----------------------------------------------------------
if numeric_for_ordinals:
    try:
        graph_dir = "../graph_emb"
        his_to_graph_path = os.path.join(graph_dir, f"task_{TASK_ID}_his_to_graph.json")

        if os.path.exists(his_to_graph_path):
            his_to_graph = json.load(open(his_to_graph_path))

            # Load graph edges from module used during embedding creation
            from extention import compute_graph_emb_generic_npy as graph_mod
            edge_index = graph_mod.edge_index  # torch.tensor with [2, num_edges]

            # Build NetworkX graph to compute degrees
            G = nx.Graph()
            for src, dst in edge_index.t().tolist():
                G.add_edge(src, dst)

            # Threshold to define "high-degree" vs "low-degree" profiles
            degree_threshold = 5
            high_deg_idxs, low_deg_idxs = [], []
            for idx, profile in enumerate(profiles):
                # ✅ Safe degree calculation
                deg_sum = 0
                for hid in profile["his_id"]:
                    node_id = his_to_graph.get(str(hid))
                    if node_id is not None and node_id in G:
                        deg_sum += G.degree(node_id)
                    else:
                        deg_sum += 0

                if deg_sum >= degree_threshold:
                    high_deg_idxs.append(idx)
                else:
                    low_deg_idxs.append(idx)


            # Accuracy for the two groups
            high_acc = accuracy_score([refs_int[i] for i in high_deg_idxs],
                                      [preds_int[i] for i in high_deg_idxs]) if high_deg_idxs else 0.0
            low_acc = accuracy_score([refs_int[i] for i in low_deg_idxs],
                                     [preds_int[i] for i in low_deg_idxs]) if low_deg_idxs else 0.0

            graph_summary = f"""
            📈 Graph-degree breakdown:
            High-degree (>= {degree_threshold} connections) — Acc: {high_acc:.4f}, Count: {len(high_deg_idxs)}
            Low-degree (< {degree_threshold} connections) — Acc: {low_acc:.4f}, Count: {len(low_deg_idxs)}
            """
            print(graph_summary)
            write_to_file(graph_summary)
        else:
            print("⚠️ his_to_graph.json not found — skipping graph-degree breakdown.")
    except Exception as e:
        print(f"⚠️ Graph-aware metric calc failed: {e}")
else:
    print("⚠️ Graph-degree breakdown skipped — labels non-numeric.")

# 1️⃣2️⃣ Detailed classification report (silent on undefined metrics)
report = "🔍 Detailed classification report:\n" + classification_report(
    refs_int, preds_int, digits=4, zero_division=0  # zero_division avoids warnings for missing classes
)
print(report)
write_to_file(report)
# 🔟 Confusion matrix
labels_sorted = sorted(set(refs_int) | set(preds_int))
cm = confusion_matrix(refs_int, preds_int, labels=labels_sorted)

cm_text = f"🔍 Confusion Matrix:\nLabels: {labels_sorted}\n{cm}"
print(cm_text)
write_to_file(cm_text)

# Optional: Pandas table
try:
    import pandas as pd
    df_cm = pd.DataFrame(cm, index=[f"True_{l}" for l in labels_sorted], columns=[f"Pred_{l}" for l in labels_sorted])
    pandas_text = "\nConfusion Matrix (Pandas view):\n" + df_cm.to_string()
    print(pandas_text)
    write_to_file(pandas_text)
except ImportError:
    pass

write_to_file("\n=== End of Metrics ===\n")

print(f"✅ Metrics saved to: {OUTPUT_FILE}")
