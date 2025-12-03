
import os
import re
import glob
from tabulate import tabulate

def extract_metrics_from_log(log_path):
    """
    Extract MAE and RMSE values from a run log file.
    Returns tuple (mae, rmse) or (None, None) if not found.
    """
    mae, rmse = None, None
    if not os.path.exists(log_path):
        return mae, rmse

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # HuggingFace trainer prints metrics as: {"mae": ..., "rmse": ...}
            if '"mae"' in line or "mae" in line.lower():
                m_mae = re.search(r'"mae"\s*:\s*([\d\.]+)', line)
                m_rmse = re.search(r'"rmse"\s*:\s*([\d\.]+)', line)
                if m_mae:
                    mae = float(m_mae.group(1))
                if m_rmse:
                    rmse = float(m_rmse.group(1))
    return mae, rmse

def collect_results(pattern):
    """
    Collect results from logs matching given pattern.
    Example: './output_*.log'
    Returns: {task_id: (mae, rmse)}
    """
    results = {}
    for log_file in glob.glob(pattern):
        # Extract task_id from filename
        m = re.search(r"_(\d+)\.log", log_file)
        if not m:
            continue
        task_id = int(m.group(1))
        mae, rmse = extract_metrics_from_log(log_file)
        results[task_id] = (mae, rmse)
    return results

if __name__ == "__main__":
    # Baseline logs
    baseline_results = collect_results("output_*.log")
    # Slim logs
    slim_results = collect_results("output_slim_*.log")

    # Prepare table
    table_data = []
    for task_id in sorted(set(baseline_results.keys()) | set(slim_results.keys())):
        base_mae, base_rmse = baseline_results.get(task_id, (None, None))
        slim_mae, slim_rmse = slim_results.get(task_id, (None, None))
        mae_diff = None
        rmse_diff = None
        if base_mae is not None and slim_mae is not None:
            mae_diff = ((base_mae - slim_mae) / base_mae) * 100.0
        if base_rmse is not None and slim_rmse is not None:
            rmse_diff = ((base_rmse - slim_rmse) / base_rmse) * 100.0

        table_data.append([
            task_id,
            base_mae, base_rmse,
            slim_mae, slim_rmse,
            mae_diff, rmse_diff
        ])

    headers = [
        "Task",
        "Baseline MAE", "Baseline RMSE",
        "Slim MAE", "Slim RMSE",
        "% MAE Improvement", "% RMSE Improvement"
    ]
    print(tabulate(table_data, headers=headers, floatfmt=".4f"))
