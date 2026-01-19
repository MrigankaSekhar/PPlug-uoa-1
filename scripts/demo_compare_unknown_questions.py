"""
Compare outputs of personalized PersonaLLM_Slim_GNN vs plain T5 on a list of custom/unknown questions.

Usage:
  python demo_compare_unknown_questions.py
"""

import os
import sys
import torch
import json
import torch.nn.functional as F
# Allow imports from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from extention.ModelForPer_slim_GNN import PersonalLLM_Slim
from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoModel
from extention.ModelForPer_slim_GNN import PersonalLLM_Slim

# --- CONFIG ---
TASK_ID = 3
CHECKPOINT_NUM = 687
PERSONA_USER_IDX = 7  # Change this to pick a different persona from your dev set

# --- Paths ---
LLM_PATH = "../FlanT5-base"
BGE_PATH = "../bge-base-en-v1.5"
CKPT_PATH = f"../extention/output_{TASK_ID}/checkpoint-{CHECKPOINT_NUM}/pytorch_model.bin"
PROFILE_PATH = "../LaMP_time_3_subset/dev_questions.json"
IDMAP_PATH = "../bge_emb/task_3_dev_bge_idmap.json"

# --- Load models ---
llm_tokenizer = AutoTokenizer.from_pretrained(LLM_PATH, use_fast=False)
emb_tokenizer = AutoTokenizer.from_pretrained(BGE_PATH)
llm_model = T5ForConditionalGeneration.from_pretrained(LLM_PATH)
plain_llm_model = T5ForConditionalGeneration.from_pretrained(LLM_PATH)
emb_model = AutoModel.from_pretrained(BGE_PATH)

state_dict = torch.load(CKPT_PATH, map_location="cpu")
llm_model.load_state_dict(state_dict, strict=False)

model = PersonalLLM_Slim(
    llm_model=llm_model,
    emb_model=emb_model,
    max_input_len=256,
    max_new_len=32,
    task_id=TASK_ID,
    use_gate=True,
)
model.eval()

# --- Load persona profile ---
with open(PROFILE_PATH) as f:
    profiles = json.load(f)
with open(IDMAP_PATH) as f:
    his_id_to_row = json.load(f)

persona_entry = profiles[PERSONA_USER_IDX]
persona_his_ids = []
for his in persona_entry.get("profile", []):
    row_idx = his_id_to_row.get(str(his.get("id", 0)), 0)
    persona_his_ids.append(int(row_idx))
his_id_real = torch.tensor(persona_his_ids, dtype=torch.long).unsqueeze(0)

# Load graph cache for node info (if available)
import pickle
GRAPH_CACHE_PATH = "../graph_emb/task_3_graph_cache.pkl"
graph_data = None
if os.path.exists(GRAPH_CACHE_PATH):
    with open(GRAPH_CACHE_PATH, "rb") as f:
        graph_data = pickle.load(f)
    node_texts = graph_data.get("node_texts", {})
    his_to_graph = graph_data.get("his_to_graph", {})
else:
    node_texts = {}
    his_to_graph = {}

print(f"\n=== User {PERSONA_USER_IDX} Profile (Graph Diagnostics) ===")
profile_list = persona_entry.get("profile", [])
if profile_list:
    for i, his in enumerate(profile_list):
        hid = str(his.get("id", 0))
        graph_node_id = his_to_graph.get(hid, None)
        review_text = his.get("text", "(no text)")
        node_text = node_texts.get(graph_node_id, "(no graph text)") if graph_node_id is not None else "(no graph node)"
        # Try to extract sentiment/popularity from node_text (if present)
        sentiment = None
        popularity = None
        # You can add logic here if you want to parse node_text for sentiment/popularity
        print(f"  [{i}] Review ID: {hid} | Graph Node: {graph_node_id} | Text: {review_text[:80]}...")
        print(f"       Graph Node Text: {node_text[:80]}...")
else:
    print("  (No profile history found for this user)")
print("="*60)

# --- List of unknown questions ---
unknown_questions = [
    "What is the score of the following review on a scale of 1 to 5? review: This product exceeded my expectations and works flawlessly.",
    "How would you rate the following? review: The book was a bit slow and not very engaging.",
    "Give a rating for this: review: The headphones are comfortable but the sound quality is average.",
    "Rate this review: The service was terrible and I would not recommend this place.",
    "What score would you give? review: The movie was fun, but the ending was predictable.",
]

graph_questions = [
    "Based on their profile history, which does this user prefer: 'Electronics' or 'Books'?",
    "Given users with similar review history, what rating would this user likely give to this item?",
    "Would this user purchase this item? Answer yes or no. Review: Durable and matches previous purchases.",
    "Is the sentiment of this review similar to those the user has written before?",
    "Does this user tend to rate popular items higher than less popular ones?",
]

unknown_questions += graph_questions

print(f"\n=== Comparing outputs for {len(unknown_questions)} unknown questions ===\n")

for idx, question in enumerate(unknown_questions):
    print(f"\n--- Question {idx+1} ---")
    print(f"Q: {question}")

    llm_input = llm_tokenizer(question, return_tensors="pt", truncation=True, max_length=256)
    emb_input = emb_tokenizer(question, return_tensors="pt", truncation=True, max_length=256)

    # --- Personalized output ---
    with torch.no_grad():
        try:
            _, seqs = model.forward(
                llm_input_ids=llm_input["input_ids"],
                llm_attention_mask=llm_input["attention_mask"],
                labels=None,
                emb_input_ids=emb_input["input_ids"],
                emb_attention_mask=emb_input["attention_mask"],
                emb_token_type_ids=torch.zeros_like(emb_input["input_ids"]),
                his_id=his_id_real
            )
        except AttributeError as e:
            if "'NoneType' object has no attribute 'long'" in str(e):
                dummy_labels = torch.zeros_like(llm_input["input_ids"])
                _, seqs = model.forward(
                    llm_input_ids=llm_input["input_ids"],
                    llm_attention_mask=llm_input["attention_mask"],
                    labels=dummy_labels,
                    emb_input_ids=emb_input["input_ids"],
                    emb_attention_mask=emb_input["attention_mask"],
                    emb_token_type_ids=torch.zeros_like(emb_input["input_ids"]),
                    his_id=his_id_real
                )
            else:
                raise
        persona_text = llm_tokenizer.decode(seqs[0], skip_special_tokens=True).strip()
        if persona_text == "" or set(persona_text) == {"."}:
            persona_text = "(no meaningful output)"

        # Diagnostics: Cosine similarity and gate value
        task_embs = model.obtain_task_emb(
            emb_input_ids=emb_input["input_ids"],
            emb_attention_mask=emb_input["attention_mask"],
            emb_token_type_ids=torch.zeros_like(emb_input["input_ids"])
        )
        persona_embs = model.obtain_profile_emb(his_id_real, task_embs)
        cos_sim = F.cosine_similarity(task_embs, persona_embs).item()
        gate_in = torch.cat([task_embs, persona_embs], dim=-1)
        gate_out = torch.sigmoid(model.gate(gate_in))
        gate_val = gate_out.mean().item()
        print(f"[Diag] Cosine similarity (task/persona): {cos_sim:.4f}")
        print(f"[Diag] Gate value: {gate_val:.4f}")

    # --- Plain T5 output ---
    with torch.no_grad():
        baseline_output_ids = plain_llm_model.generate(
            input_ids=llm_input["input_ids"],
            attention_mask=llm_input["attention_mask"],
            max_new_tokens=32,
            num_beams=4,
            do_sample=False
        )
        plain_text = llm_tokenizer.decode(baseline_output_ids[0], skip_special_tokens=True).strip()
        if plain_text == "" or set(plain_text) == {"."}:
            plain_text = "(no meaningful output)"

    print(f"Personalized → {persona_text}")
    print(f"Plain T5     → {plain_text}")

print("\n=== Done ===")