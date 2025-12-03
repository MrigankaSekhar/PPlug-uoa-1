
import os
import math
import transformers
from dataclasses import dataclass, field
from transformers import (
    T5Tokenizer,
    AutoTokenizer, AutoModel,
    Seq2SeqTrainer, Seq2SeqTrainingArguments
)
from PersonalDataset_profile import PersonalDataset
from ModelForPer_slim_bnb import PersonalLLM_Slim_BNB   # <-- new slim+B&A model

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

    # Model — new Slim variant with optional quantization
    task_id = int(training_args.output_dir.split("_")[-1])
    model = PersonalLLM_Slim_BNB(
        model_args=model_args,
        emb_model=emb_model,
        max_input_len=data_args.max_input_len,
        max_new_len=data_args.max_new_len,
        task_id=task_id,
        use_4bit=model_args.use_4bit,
        use_8bit=model_args.use_8bit
    )

    model.llm_model.resize_token_embeddings(len(llm_tokenizer))

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=llm_tokenizer,
        compute_metrics=compute_metrics_classification
    )

    # Train
    trainer.train()
    outputs = trainer.evaluate()
    print(outputs)

# -----------------------------
# Entry Point
# -----------------------------
if __name__ == '__main__':
    transformers.set_seed(42)
    global llm_tokenizer, emb_tokenizer

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    train_model(model_args, data_args, training_args)
