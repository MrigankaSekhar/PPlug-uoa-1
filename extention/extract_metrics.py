#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description="Extract final eval metrics from a HuggingFace trainer_state.json")
    p.add_argument("--state-file", required=True, help="Path to trainer_state.json")
    p.add_argument("--exp-name", required=True, help="Experiment name")
    p.add_argument("--duration-sec", required=True, type=int, help="Training duration in seconds")
    p.add_argument("--out-file", required=True, help="Where to write extracted metrics JSON")
    args = p.parse_args()

    state_file = Path(args.state_file)
    if not state_file.exists():
        print(f"❌ state_file not found: {state_file}", file=sys.stderr)
        return 1

    try:
        with state_file.open("r") as f:
            state = json.load(f)

        log_history = state.get("log_history", [])

        eval_metrics = None
        for entry in reversed(log_history):
            if "eval_mae" in entry:
                eval_metrics = entry
                break

        if eval_metrics is None:
            print(f"⚠️  No eval metrics found for {args.exp_name}", file=sys.stderr)
            return 1

        result = {
            "experiment": args.exp_name,
            "mae": eval_metrics.get("eval_mae", None),
            "rmse": eval_metrics.get("eval_rmse", None),
            "accuracy": eval_metrics.get("eval_rating_acc", None),
            "n_samples": eval_metrics.get("eval_n_rating", 0),
            "n_total": eval_metrics.get("eval_n_total", 0),
            "training_time_sec": args.duration_sec,
            "training_time_min": round(args.duration_sec / 60, 1),
            "final_loss": eval_metrics.get("eval_loss", None),
        }

        out_path = Path(args.out_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(result, f, indent=2)

        print(f"✅ Extracted metrics for {args.exp_name}")
        if result["mae"] is not None and result["rmse"] is not None and result["accuracy"] is not None:
            print(f"   MAE={result['mae']:.3f}, RMSE={result['rmse']:.3f}, Acc={result['accuracy']:.3f}")
        return 0

    except Exception as e:
        print(f"❌ Error extracting metrics for {args.exp_name}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())