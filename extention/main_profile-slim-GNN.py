
import os
import math
import transformers
from dataclasses import dataclass, field
from transformers import (
    T5Tokenizer,
    AutoTokenizer, AutoModel,
    Seq2SeqTrainer, Seq2SeqTrainingArguments
)

# IMPORT from the extension folder
from PersonalDataset_profile_GNN import PersonalDataset
from ModelForPer_slim_GNN import PersonalLLM_Slim   # <-- updated slim model

# -----------------------------
# Arguments Setup
# -----------------------------
@dataclass
class ModelArguments:
    model_path: str = field(default="../flant5-small/", metadata={"help": "Path to the pretrain model."})
    emb_model_path: str = field(default="../bge-base-en-v1.5/", metadata={"help": "Path to the embedding model"})
    use_4bit: bool = field(default=False, metadata={"help": "Enable 4-bit BitsAndBytes quantization"})
    use_8bit: bool = field(default=False, metadata={"help": "Enable 8-bit BitsAndBytes quantization"})

@dataclass
class DataArguments:
    train_file: str = field(default="../data/train.json", metadata={"help": "Training dataset file path"})
    dev_file: str = field(default="../data/dev.json", metadata={"help": "Validation dataset file path"})
    max_input_len: int = field(default=256, metadata={"help": "Max input length"})
    max_new_len: int = field(default=32, metadata={"help": "Max new token length"})
    max_his_len: int = field(default=512, metadata={"help": "Max history length"})

# -----------------------------
# Metrics Functions
# -----------------------------
def compute_metrics_classification(eval_preds):
    preds, labels = eval_preds
    preds = [[max(0, idx) for idx in x] for x in preds]
    labels = [[max(0, idx) for idx in x] for x in labels]
    predictions = llm_tokenizer.batch_decode(preds, skip_special_tokens=True)
    references = llm_tokenizer.batch_decode(labels, skip_special_tokens=True)

    def create_mapping(x):
        try:
            return float(x)
        except:
            for z in x:
                if z.isnumeric():
                    return float(int(z))
            print(x)
            return 1.0

    predictions = [create_mapping(y) for y in predictions]
    references = [create_mapping(y) for y in references]

    mae = sum(abs(p - r) for p, r in zip(predictions, references)) / len(predictions)
    rmse = math.sqrt(sum((p - r) ** 2 for p, r in zip(predictions, references)) / len(predictions))
    return {"mae": mae, "rmse": rmse}

# -----------------------------
# Train Function
# -----------------------------
def train_model(model_args, data_args, training_args):
    # Tokenizers
    llm_tokenizer = T5Tokenizer.from_pretrained(model_args.model_path)
    emb_tokenizer = AutoTokenizer.from_pretrained(model_args.emb_model_path)

    # Embedding model
    emb_model = AutoModel.from_pretrained(model_args.emb_model_path)

    # Dataset
    train_dataset = PersonalDataset(
        data_args.train_file, data_args.max_input_len, data_args.max_new_len,
        data_args.max_his_len, llm_tokenizer, emb_tokenizer
    )
    eval_dataset = PersonalDataset(
        data_args.dev_file, data_args.max_input_len, data_args.max_new_len,
        data_args.max_his_len, llm_tokenizer, emb_tokenizer
    )

    # Model — our updated slim variant
    task_id = int(training_args.output_dir.split("_")[-1])
    model = PersonalLLM_Slim(
        llm_model=transformers.T5ForConditionalGeneration.from_pretrained(model_args.model_path),
        emb_model=emb_model,
        max_input_len=data_args.max_input_len,
        max_new_len=data_args.max_new_len,
        task_id=task_id
    )
    model.llm_model.resize_token_embeddings(len(llm_tokenizer))

    # -----------------------------
    # Custom collator
    # -----------------------------
    import torch
    def my_collator(features):
        """Convert lists of NumPy arrays to stacked tensors with fixed shapes."""
        return {
            'llm_input_ids': torch.stack([torch.as_tensor(f['llm_input_ids'], dtype=torch.long) for f in features]),
            'llm_attention_mask': torch.stack([torch.as_tensor(f['llm_attention_mask'], dtype=torch.long) for f in features]),
            'labels': torch.stack([torch.as_tensor(f['labels'], dtype=torch.long) for f in features]),
            'emb_input_ids': torch.stack([torch.as_tensor(f['emb_input_ids'], dtype=torch.long) for f in features]),
            'emb_attention_mask': torch.stack([torch.as_tensor(f['emb_attention_mask'], dtype=torch.long) for f in features]),
            'emb_token_type_ids': torch.stack([torch.as_tensor(f['emb_token_type_ids'], dtype=torch.long) for f in features]),
            'his_id': torch.stack([torch.as_tensor(f['his_id'], dtype=torch.long) for f in features]),
            'session_ids': torch.stack([torch.as_tensor(f['session_ids'], dtype=torch.long) for f in features]),       # NEW
            'graph_node_ids': torch.stack([torch.as_tensor(f['graph_node_ids'], dtype=torch.long) for f in features]) # NEW
        }

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=llm_tokenizer,
        data_collator=my_collator,
        compute_metrics=compute_metrics_classification
    )

    # Train & evaluate
    trainer.train()
    outputs = trainer.evaluate()
    print(outputs)

# -----------------------------
# Entry Point
# -----------------------------
if __name__ == '__main__':
    transformers.set_seed(42)
    global llm_tokenizer, emb_tokenizer

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    train_model(model_args, data_args, training_args)
