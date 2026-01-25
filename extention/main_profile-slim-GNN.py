
import os
import math
import transformers
from dataclasses import dataclass, field
from transformers import (
    T5Tokenizer,
    AutoTokenizer,
    AutoModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)

# IMPORT from the extension folder
from PersonalDataset_profile_GNN import PersonalDataset
from ModelForPer_slim_GNN import PersonalLLM_Slim   # <-- updated slim model
from torch.utils.data import Subset
from torch.nn.utils.rnn import pad_sequence
import json
from transformers import TrainerCallback
import torch

class ConsoleMetricsCallback(TrainerCallback):
    """
    A callback that prints selected metrics to stdout in a JSON-like dict form
    every time logging happens.
    """
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        keys_of_interest = ['loss', 'grad_norm', 'learning_rate', 'epoch', 'mae', 'rmse']
        selected = {k: logs[k] for k in keys_of_interest if k in logs}
        if selected:
            print(f"[METRICS] {json.dumps(selected)}")


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
    use_subset: bool = field(default=True, metadata={"help": "Use a smaller subset for quick testing (default=True)"})


# -----------------------------
# Train Function
# -----------------------------
def train_model(model_args, data_args, training_args):

    llm_tokenizer = T5Tokenizer.from_pretrained(model_args.model_path)
    llm_tokenizer.model_max_length = data_args.max_input_len
    emb_tokenizer = AutoTokenizer.from_pretrained(model_args.emb_model_path)
    emb_tokenizer.model_max_length = data_args.max_input_len

    llm_model_loaded = transformers.T5ForConditionalGeneration.from_pretrained(model_args.model_path)

    # ✅ Force config limits to match dataset cropping for no token-length warning
    llm_model_loaded.config.n_positions = data_args.max_input_len
    llm_model_loaded.config.max_position_embeddings = data_args.max_input_len
    llm_model_loaded.config.model_max_length = data_args.max_input_len


    def compute_metrics_classification(eval_preds):
        preds, labels = eval_preds
        preds = [[max(0, idx) for idx in x] for x in preds]
        labels = [[max(0, idx) for idx in x] for x in labels]
        predictions = llm_tokenizer.batch_decode(preds, skip_special_tokens=True)
        references = llm_tokenizer.batch_decode(labels, skip_special_tokens=True)

        # 🔹 DEBUG LOGGING
        print("\n[DEBUG] Evaluation Predictions vs References")
        for i, (p, r) in enumerate(zip(predictions[:10], references[:10])):
            print(f"Sample {i}: PRED='{p}' | REF='{r}'")

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

    emb_model = AutoModel.from_pretrained(model_args.emb_model_path)

    # Dataset building
    graph_emb_npy_path = f"../graph_emb/task_{int(training_args.output_dir.split('_')[-1])}_graph.npy"
    his_to_graph_path = f"../graph_emb/task_{int(training_args.output_dir.split('_')[-1])}_his_to_graph.json"

    train_dataset = PersonalDataset(
        data_args.train_file, data_args.max_input_len, data_args.max_new_len,
        data_args.max_his_len, llm_tokenizer, emb_tokenizer,
        graph_emb_path=graph_emb_npy_path,
        his_to_graph_path=his_to_graph_path
    )
    eval_dataset = PersonalDataset(
        data_args.dev_file, data_args.max_input_len, data_args.max_new_len,
        data_args.max_his_len, llm_tokenizer, emb_tokenizer,
        graph_emb_path=graph_emb_npy_path,
        his_to_graph_path=his_to_graph_path
    )

    task_id = int(training_args.output_dir.split("_")[-1])


    model = PersonalLLM_Slim(
        llm_model=llm_model_loaded,
        emb_model=emb_model,
        max_input_len=data_args.max_input_len,
        max_new_len=data_args.max_new_len,
        task_id=task_id
    )
    model.llm_model.resize_token_embeddings(len(llm_tokenizer))

    def my_collator(features):
        def pad_to_longest(key, dtype=torch.long, pad_value=0):
            return pad_sequence(
                [torch.as_tensor(f[key], dtype=dtype) for f in features],
                batch_first=True,
                padding_value=pad_value
            )
        return {
            'llm_input_ids': pad_to_longest('llm_input_ids'),
            'llm_attention_mask': pad_to_longest('llm_attention_mask'),
            'labels': pad_to_longest('labels'),
            'emb_input_ids': pad_to_longest('emb_input_ids'),
            'emb_attention_mask': pad_to_longest('emb_attention_mask'),
            'emb_token_type_ids': pad_to_longest('emb_token_type_ids'),
            'his_id': pad_to_longest('his_id'),
            'session_ids': pad_to_longest('session_ids'),
            'graph_node_ids': pad_to_longest('graph_node_ids')
        }

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=llm_tokenizer,
        data_collator=my_collator,
        compute_metrics=compute_metrics_classification,
        callbacks=[ConsoleMetricsCallback]
    )

    trainer.train()
    outputs = trainer.evaluate()
    print(outputs)


# -----------------------------
# Entry Point
# -----------------------------
if __name__ == '__main__':
    transformers.set_seed(42)
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.save_safetensors = False
    train_model(model_args, data_args, training_args)