import torch
import json
from transformers import T5ForConditionalGeneration, AutoModel, AutoTokenizer
from code.ModelForPer import PersonalLLM  # import your existing class

# --------------------------
# CONFIGURATION
# --------------------------
TASK_ID = 3
BASE_MODEL_PATH = "../FlanT5-small"
EMB_MODEL_PATH = "../bge-base-en-v1.5"
CKPT_DIR = f"output_{TASK_ID}"
BGE_DIR = "./bge_emb"
DEV_FILE = f"../LaMP_time_{TASK_ID}_id/dev_profile.json"

MAX_INPUT_LEN = 256
MAX_HIS_LEN = 512
MAX_NEW_LEN = 10

device = torch.device("cpu")  # CPU only

# --------------------------
# LOAD MODELS
# --------------------------
print("Loading base model & tokenizer...")
llm_model = T5ForConditionalGeneration.from_pretrained(BASE_MODEL_PATH).to(device)
llm_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

print("Loading embedding model...")
emb_model = AutoModel.from_pretrained(EMB_MODEL_PATH).to(device)
emb_tokenizer = AutoTokenizer.from_pretrained(EMB_MODEL_PATH)

print("Loading trained adapter weights...")
llm_model.load_state_dict(
    torch.load(f"{CKPT_DIR}/pytorch_model.bin", map_location=device),
    strict=False
)

# Wrap in PersonalLLM to reuse profile embedding logic
print("Initializing PersonalLLM...")
model = PersonalLLM(llm_model, emb_model, MAX_INPUT_LEN, MAX_NEW_LEN, TASK_ID).to(device)
model.eval()

# --------------------------
# LOAD DEV DATA
# --------------------------
print("Loading dev set...")
with open(DEV_FILE, "r") as f:
    dev_data = json.load(f)

# --------------------------
# RUN INFERENCE
# --------------------------
for sample in dev_data[:5]:  # run on first 5 samples for test
    # Prepare LLM inputs
    llm_input = llm_tokenizer(
        sample["input_text"],  # assuming dev_profile.json has "input_text"
        return_tensors="pt",
        max_length=MAX_INPUT_LEN,
        truncation=True
    )

    # Prepare embedding model inputs
    emb_input = emb_tokenizer(
        sample["input_text"],
        return_tensors="pt",
        max_length=MAX_INPUT_LEN,
        truncation=True
    )

    # Prepare history IDs (his_id) from sample["history_ids"]
    his_id = torch.tensor(sample["history_ids"]).unsqueeze(0)  # batch size 1

    with torch.no_grad():
        output = model.forward(
            llm_input_ids=llm_input["input_ids"],
            llm_attention_mask=llm_input["attention_mask"],
            labels=None,
            emb_input_ids=emb_input["input_ids"],
            emb_attention_mask=emb_input["attention_mask"],
            emb_token_type_ids=torch.zeros_like(emb_input["input_ids"]),  # placeholder
            his_id=his_id
        )

        # In eval branch, forward() returns: [loss, sequences]
        _, sequences = output
        pred_text = llm_tokenizer.decode(sequences[0], skip_special_tokens=True)

    print(f"Q: {sample['input_text']}")
    print(f"Predicted rating: {pred_text}\n")