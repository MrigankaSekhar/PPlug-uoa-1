
import os
import math
import transformers
from dataclasses import dataclass, field
from typing import Optional
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
from eval_logging import (
        extract_rating_1_to_5, open_jsonl, log_eval_row
    )
from torch.utils.data import Subset
from torch.nn.utils.rnn import pad_sequence
import json
from transformers import TrainerCallback
import torch
import re

def _get_allowed_rating_token_ids(tokenizer):
    """
    Restrict decoding to a single token among: '1','2','3','4','5'
    If the tokenizer splits these into multiple tokens (rare for T5), we fall back gracefully.
    """
    allowed = []
    for s in ["1", "2", "3", "4", "5"]:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if len(ids) == 1:
            allowed.append(ids[0])
    return allowed

class _RestrictToRatingTokensProcessor(transformers.LogitsProcessor):
    def __init__(self, allowed_token_ids):
        super().__init__()
        self.allowed = set(allowed_token_ids)

    def __call__(self, input_ids, scores):
        if not self.allowed:
            return scores
        mask = torch.full_like(scores, float("-inf"))
        for tid in self.allowed:
            mask[:, tid] = 0.0
        return scores + mask

class RatingOnlyLogitsProcessorList(transformers.LogitsProcessorList):
    pass

# --- PATCH: Optional per-step fusion/data diagnostics (enable with PPLOG_DIAG=1) ---
def _safe_ratio(x: torch.Tensor) -> float:
    if x is None:
        return float("nan")
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    if x.numel() == 0:
        return float("nan")
    return float(x.float().mean().detach().cpu())

class FusionDataDiagnosticsCallback(TrainerCallback):
    """
    Prints lightweight diagnostics about the collated batch so we can verify:
      - session_ids are not mostly padding
      - graph_node_mask has valid nodes
      - graph_node_ids contain non -1 values
      - his_id is present

    Enable with:
      export PPLOG_DIAG=1
    Control frequency with:
      export PPLOG_DIAG_EVERY=50
    """
    def __init__(self):
        super().__init__()
        self.enabled = os.environ.get("PPLOG_DIAG", "").strip().lower() in {"1", "true", "yes", "y"}
        self.every = int(os.environ.get("PPLOG_DIAG_EVERY", "50"))

    def on_train_batch_begin(self, args, state, control, **kwargs):
        if not self.enabled:
            return
        if self.every <= 0:
            return
        if state.global_step % self.every != 0:
            return

        inputs = kwargs.get("inputs", None)
        if inputs is None:
            return

        with torch.no_grad():
            out = {"step": int(state.global_step)}

            # session_ids: assume padding is 0 (per collator)
            if "session_ids" in inputs and inputs["session_ids"] is not None:
                sid = inputs["session_ids"]
                out["session_nonpad_ratio"] = _safe_ratio(sid.ne(0))

            # his_id: assume padding is 0 (per collator)
            if "his_id" in inputs and inputs["his_id"] is not None:
                hid = inputs["his_id"]
                out["his_nonpad_ratio"] = _safe_ratio(hid.ne(0))

            # graph_node_mask: should be 0/1
            if "graph_node_mask" in inputs and inputs["graph_node_mask"] is not None:
                gmask = inputs["graph_node_mask"]
                out["graph_mask_mean"] = _safe_ratio(gmask)

            # graph_node_ids: padding is -1 (per collator)
            if "graph_node_ids" in inputs and inputs["graph_node_ids"] is not None:
                gids = inputs["graph_node_ids"]
                out["graph_id_valid_ratio"] = _safe_ratio(gids.ne(-1))

            print(f"[DIAG] {json.dumps(out)}")
# --- END PATCH ---

def extract_yes_no(text: str):
    """Extract yes/no with word boundaries."""
    if text is None:
        return None
    text_lower = str(text).lower()
    if re.search(r'\byes\b', text_lower):
        return "yes"
    if re.search(r'\bno\b', text_lower):
        return "no"
    return None

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

    # --- NEW: personalization source toggles ---
    use_profile: bool = field(default=True, metadata={"help": "Use long-term profile/history embeddings (his_id)."})
    use_session: bool = field(default=True, metadata={"help": "Use short-term session embeddings (session_ids)."})
    use_graph: bool = field(default=True, metadata={"help": "Use graph embeddings (graph_node_ids)."})
    use_inst_token: bool = field(default=False, metadata={"help": "Enable trainable instruction token injection."})

@dataclass
class DataArguments:
    train_file: str = field(default="../data/train.json", metadata={"help": "Training dataset file path"})
    dev_file: str = field(default="../data/dev.json", metadata={"help": "Validation dataset file path"})
    max_input_len: int = field(default=256, metadata={"help": "Max input length"})
    max_new_len: int = field(default=32, metadata={"help": "Max new token length"})
    max_his_len: int = field(default=512, metadata={"help": "Max history length"})
    use_subset: bool = field(default=True, metadata={"help": "Use a smaller subset for quick testing (default=True)"})
    max_session_len: int = field(default=7, metadata={"help": "Maximum number of recent session items to use"})
    task_id: Optional[int] = field(default=None, metadata={"help": "LaMP task ID (1-7). Required for dataset loading."})


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

        pred_texts = llm_tokenizer.batch_decode(preds, skip_special_tokens=True)
        gold_texts = llm_tokenizer.batch_decode(labels, skip_special_tokens=True)

        # -----------------------------
        # JSONL eval logging (optional)
        # Enable by setting env var:
        #   export PPLOG_EVAL_JSONL="../logs/eval_debug.jsonl"
        # -----------------------------
        log_path = os.environ.get("PPLOG_EVAL_JSONL", "").strip()
        log_fp = open_jsonl(log_path) if log_path else None

        rating_preds, rating_golds = [], []
        yn_correct = 0
        yn_total = 0
        skipped = 0

        # Debug print (first 20)
        print("\n[DEBUG] Eval pred vs gold (first 20)")
        for i, (p, g) in enumerate(zip(pred_texts, gold_texts)):
            if i >= 20:
                break
            pr = extract_rating_1_to_5(p)
            gr = extract_rating_1_to_5(g)
            py = extract_yes_no(p)
            gy = extract_yes_no(g)
            print(f"Sample {i}: pred='{p}' (rating={pr}, yn={py}) | gold='{g}' (rating={gr}, yn={gy})")

        for i, (p, g) in enumerate(zip(pred_texts, gold_texts)):
            pr = extract_rating_1_to_5(p)
            gr = extract_rating_1_to_5(g)
            if pr is not None and gr is not None:
                rating_preds.append(float(pr))
                rating_golds.append(float(gr))

                if log_fp is not None:
                    log_eval_row(log_fp, {
                        "i": i,
                        "type": "rating",
                        "pred_text": str(p),
                        "gold_text": str(g),
                        "pred_rating": pr,
                        "gold_rating": gr,
                        "correct": bool(int(pr) == int(gr)),
                    })
                continue

            py = extract_yes_no(p)
            gy = extract_yes_no(g)
            if py is not None and gy is not None:
                yn_total += 1
                if py == gy:
                    yn_correct += 1

                if log_fp is not None:
                    log_eval_row(log_fp, {
                        "i": i,
                        "type": "yesno",
                        "pred_text": str(p),
                        "gold_text": str(g),
                        "pred_yesno": py,
                        "gold_yesno": gy,
                        "correct": bool(py == gy),
                    })
                continue

            skipped += 1
            if log_fp is not None:
                log_eval_row(log_fp, {
                    "i": i,
                    "type": "skipped",
                    "pred_text": str(p),
                    "gold_text": str(g),
                    "pred_rating": pr,
                    "gold_rating": gr,
                    "pred_yesno": py,
                    "gold_yesno": gy,
                })

        if log_fp is not None:
            log_fp.close()

        metrics = {"n_total": len(pred_texts), "n_skipped": skipped}

        # Rating metrics (regression)
        if rating_preds:
            n = len(rating_preds)
            mae = sum(abs(p - r) for p, r in zip(rating_preds, rating_golds)) / n
            rmse = math.sqrt(sum((p - r) ** 2 for p, r in zip(rating_preds, rating_golds)) / n)
            acc = sum(1 for p, r in zip(rating_preds, rating_golds) if int(p) == int(r)) / n
            mape = sum(abs((p - r) / r) if r != 0 else 0.0 for p, r in zip(rating_preds, rating_golds)) / n
            metrics.update({
                "rating_acc": acc,
                "rating_mae": mae,
                "rating_rmse": rmse,
                "rating_mape": mape,
                "n_rating": n
            })
        else:
            metrics.update({"n_rating": 0})

        # Yes/No metrics (classification)
        if yn_total:
            metrics.update({
                "yn_acc": yn_correct / yn_total,
                "n_yn": yn_total
            })
        else:
            metrics.update({"n_yn": 0})

        # Backwards-compat keys so Trainer logs don't break existing plots:
        # If we have rating metrics, expose them as mae/rmse. Otherwise fall back to yn_acc.
        if metrics["n_rating"] > 0:
            metrics["mae"] = metrics["rating_mae"]
            metrics["rmse"] = metrics["rating_rmse"]
            metrics["mape"] = metrics["rating_mape"]
            metrics["acc"] = metrics["rating_acc"]
        elif metrics["n_yn"] > 0:
            metrics["acc"] = metrics["yn_acc"]

        return metrics

    emb_model = AutoModel.from_pretrained(model_args.emb_model_path)

    # Dataset building
    # Robust task_id extraction
    if data_args.task_id is not None:
        task_id = data_args.task_id
    else:
        # Try multiple fallback strategies
        import re
        
        # Strategy 1: Extract from train_file
        match = re.search(r'LaMP_time_(\d+)', data_args.train_file)
        if match:
            task_id = int(match.group(1))
        else:
            # Strategy 2: Extract from dev_file
            match = re.search(r'LaMP_time_(\d+)', data_args.dev_file)
            if match:
                task_id = int(match.group(1))
            else:
                # Strategy 3: Try output_dir (extract first digit sequence)
                match = re.search(r'task(\d+)|_(\d+)_', training_args.output_dir)
                if match:
                    task_id = int(match.group(1) or match.group(2))
                else:
                    raise ValueError(
                        "Cannot determine task_id. Please provide --task_id argument "
                        "or ensure train_file/dev_file contain 'LaMP_time_X' pattern."
                    )
    
    print(f"📋 Task ID: {task_id}")
    GRAPH_DIR = "../graph_emb"

    graph_emb_npy_path = os.path.join(GRAPH_DIR, f"task_{task_id}_graph.npy")

    mapping_candidates = [
        os.path.join(GRAPH_DIR, f"task_{task_id}_his_to_graph_node.json"),
        os.path.join(GRAPH_DIR, f"task_{task_id}_his_to_graph.json"),
    ]
    his_to_graph_path = next((p for p in mapping_candidates if os.path.exists(p)), None)

    if his_to_graph_path is None:
        raise FileNotFoundError(
            f"Missing his_to_graph mapping. Tried: {mapping_candidates}"
        )

    train_dataset = PersonalDataset(
        data_args.train_file, data_args.max_input_len, data_args.max_new_len,
        data_args.max_his_len, llm_tokenizer, emb_tokenizer,
        graph_emb_path=graph_emb_npy_path,
        his_to_graph_path=his_to_graph_path,
        max_session_len=getattr(data_args, "max_session_len", 3),
    )
    eval_dataset = PersonalDataset(
        data_args.dev_file, data_args.max_input_len, data_args.max_new_len,
        data_args.max_his_len, llm_tokenizer, emb_tokenizer,
        graph_emb_path=graph_emb_npy_path,
        his_to_graph_path=his_to_graph_path,
        max_session_len=getattr(data_args, "max_session_len", 3),
    )


    model = PersonalLLM_Slim(
        llm_model=llm_model_loaded,
        emb_model=emb_model,
        llm_tokenizer=llm_tokenizer,
        max_input_len=data_args.max_input_len,
        max_new_len=data_args.max_new_len,
        task_id=task_id,
        use_profile=model_args.use_profile,
        use_session=model_args.use_session,
        use_graph=model_args.use_graph,
        use_inst_token=model_args.use_inst_token,
    )
    model.llm_model.resize_token_embeddings(len(llm_tokenizer))

    # --- PATCH: make wrapper model compatible with Seq2SeqTrainer.generate() ---
    # Seq2SeqTrainer expects `model.generate(...)` to exist. Our wrapper keeps the actual
    # HF T5 under `model.llm_model`, so expose a generate proxy.
    if not hasattr(model, "generate") and hasattr(model, "llm_model") and hasattr(model.llm_model, "generate"):
        def generate(self, *args, **kwargs):
            return self.llm_model.generate(*args, **kwargs)
        model.generate = generate.__get__(model, model.__class__)
    # --- END PATCH ---

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
            'graph_node_ids': pad_to_longest('graph_node_ids', pad_value=-1),
            'graph_node_mask': pad_to_longest('graph_node_mask')
        }

    allowed_rating_token_ids = _get_allowed_rating_token_ids(llm_tokenizer)

    rating_logits_processors = RatingOnlyLogitsProcessorList()
    rating_logits_processors.append(_RestrictToRatingTokensProcessor(allowed_rating_token_ids))

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=llm_tokenizer,
        data_collator=my_collator,
        compute_metrics=compute_metrics_classification,
        callbacks=[ConsoleMetricsCallback, FusionDataDiagnosticsCallback()]
    )

    # --- PATCH: force rating-only decoding during evaluation/generation ---
    # HF Trainer uses model.generation_config + generation kwargs; we hook via predict/evaluate kwargs by setting on model
    try:
        trainer.model.generation_config.max_length = 2
        trainer.model.generation_config.num_beams = 5
        trainer.model.generation_config.do_sample = False
    except Exception:
        pass

    # We also monkey-patch trainer's generation logits processors via a small wrapper around model.generate
    # (minimal + local; doesn't require editing transformers internals)
    _orig_generate = trainer.model.generate
    def _rating_only_generate(*args, **kwargs):
        kwargs.setdefault("max_length", 2)
        kwargs.setdefault("num_beams", 5)
        kwargs.setdefault("do_sample", False)
        kwargs["logits_processor"] = rating_logits_processors
        return _orig_generate(*args, **kwargs)

    trainer.model.generate = _rating_only_generate
    # --- END PATCH ---

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