#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser(description="Aggregate ablation metrics and write summary + markdown table")
    p.add_argument("--metrics-dir", required=True, help="Directory with per-run metrics JSON files")
    p.add_argument("--results-file", required=True, help="Where to write aggregated JSON")
    p.add_argument("--table-file", required=True, help="Where to write markdown table")
    p.add_argument("--timestamp", default="", help="Timestamp string for table header")
    p.add_argument("--num-seeds", type=int, default=3, help="Runs per variant")
    p.add_argument("--task-id", default="3", help="Task ID for table header")
    args = p.parse_args()

    metrics_dir = Path(args.metrics_dir)
    all_metrics = []
    for mf in metrics_dir.glob("*.json"):
        with mf.open("r") as f:
            all_metrics.append(json.load(f))

    if not all_metrics:
        print("❌ No metrics found!")
        return 1

    variants = {}
    for m in all_metrics:
        variant_name = m["experiment"].split("_seed")[0]
        variants.setdefault(variant_name, []).append(m)

    results = {}
    for variant, runs in variants.items():
        mae_values = [r["mae"] for r in runs if r.get("mae") is not None]
        rmse_values = [r["rmse"] for r in runs if r.get("rmse") is not None]
        acc_values = [r["accuracy"] for r in runs if r.get("accuracy") is not None]
        time_values = [r["training_time_min"] for r in runs if r.get("training_time_min") is not None]

        results[variant] = {
            "mae_mean": float(np.mean(mae_values)) if mae_values else None,
            "mae_std": float(np.std(mae_values)) if len(mae_values) > 1 else 0.0,
            "rmse_mean": float(np.mean(rmse_values)) if rmse_values else None,
            "rmse_std": float(np.std(rmse_values)) if len(rmse_values) > 1 else 0.0,
            "acc_mean": float(np.mean(acc_values)) if acc_values else None,
            "acc_std": float(np.std(acc_values)) if len(acc_values) > 1 else 0.0,
            "time_mean": float(np.mean(time_values)) if time_values else None,
            "n_runs": len(runs),
        }

    results_path = Path(args.results_file)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Saved aggregated results to {results_path}")

    table_lines = [
        "# Ablation Study Results",
        "",
        f"**Generated:** {args.timestamp}",
        f"**Task:** LaMP-{args.task_id}",
        f"**Runs per variant:** {args.num_seeds}",
        "",
        "## Table 1: Component Ablation on LaMP-3",
        "",
        "| Model Variant | MAE ↓ | RMSE ↓ | Accuracy ↑ | Training Time | Runs |",
        "|---------------|-------|--------|------------|---------------|------|",
    ]

    variant_order = ["baseline", "session", "graph", "full"]
    variant_labels = {
        "baseline": "Profile only",
        "session": "+ Session",
        "graph": "+ Graph",
        "full": "**+ Both (ours)**",
    }

    for variant in variant_order:
        if variant not in results:
            continue

        r = results[variant]
        label = variant_labels.get(variant, variant)

        mae_str = f"{r['mae_mean']:.3f} ± {r['mae_std']:.3f}" if r["mae_mean"] is not None else "N/A"
        rmse_str = f"{r['rmse_mean']:.3f} ± {r['rmse_std']:.3f}" if r["rmse_mean"] is not None else "N/A"
        acc_str = f"{r['acc_mean']*100:.1f}%" if r["acc_mean"] is not None else "N/A"
        time_str = f"{r['time_mean']:.0f} min" if r["time_mean"] is not None else "N/A"

        if variant == "full":
            table_lines.append(f"| {label} | **{mae_str}** | **{rmse_str}** | **{acc_str}** | **{time_str}** | {r['n_runs']} |")
        else:
            table_lines.append(f"| {label} | {mae_str} | {rmse_str} | {acc_str} | {time_str} | {r['n_runs']} |")

    table_lines.extend(
        [
            "",
            "---",
            "",
            "**Notes:**",
            f"- Results averaged over {args.num_seeds} runs with different random seeds (42, 43, 44)",
            "- ↓ = Lower is better, ↑ = Higher is better",
            "- Statistical significance can be computed via paired t-test (see `compute_significance.py`)",
            "",
        ]
    )

    table_path = Path(args.table_file)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with table_path.open("w") as f:
        f.write("\n".join(table_lines))

    print(f"✅ Saved table to {table_path}")
    print("")
    print("=" * 60)
    print("📋 ABLATION STUDY COMPLETE")
    print("=" * 60)
    print("")
    print("\n".join(table_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())