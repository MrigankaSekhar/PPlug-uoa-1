
import torch
import json
from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoModel
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
import numpy as np
import sys
import os
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
CHEKPOINT_NUM=69
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
            labels=None,
            emb_input_ids=emb_input["input_ids"],
            emb_attention_mask=emb_input["attention_mask"],
            emb_token_type_ids=torch.zeros_like(emb_input["input_ids"]),
            his_id=his_id
        )
    pred_text = llm_tokenizer.decode(sequences[0], skip_special_tokens=True).strip()

    preds_str.append(pred_text)
    refs_str.append(ref)

# 7️⃣ Convert labels to integers if possible
try:
    preds_int = [int(p) for p in preds_str]
    refs_int = [int(r) for r in refs_str]
except ValueError:
    preds_int, refs_int = preds_str, refs_str

# 8️⃣ Metrics
acc = accuracy_score(refs_int, preds_int)
f1 = f1_score(refs_int, preds_int, average="macro")

summary = "\n=======================\n"
summary += f"📊 Accuracy: {acc:.4f}\n"
summary += f"📊 Macro F1: {f1:.4f}\n"
summary += "=======================\n"
print(summary)
write_to_file(summary)

# 9️⃣ Detailed classification report
report = "🔍 Detailed classification report:\n" + classification_report(refs_int, preds_int, digits=4)
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
