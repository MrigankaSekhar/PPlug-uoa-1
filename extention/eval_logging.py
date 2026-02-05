import json
import os
import re
from datetime import datetime
from typing import Optional, Tuple, Any, Dict

_YN_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)

def extract_rating_1_to_5(text: str):
    """
    Extract rating 1-5 from model output.
    Skips invalid outputs (0, empty, non-numeric).
    """
    import re
    if not text or not text.strip():
        return None
    
    # ✅ Only match digits 1-5
    match = re.search(r'[1-5]', text.strip())
    if not match:
        return None
    
    return int(match.group())

def extract_yes_no(text: Any) -> Optional[str]:
    """
    Extract yes/no from text using word boundaries.
    ✅ FIX: Prevents '1' from being detected as 'yes'
    """
    if text is None:
        return None
    
    t = str(text).strip().lower()
    
    # ✅ Use word boundaries to match only standalone 'yes'/'no'
    m = _YN_RE.search(t)
    if m:
        return m.group(1).lower()
    
    # ✅ REMOVED: Special cases for '1'/'0' that caused false positives
    # Old code had: if t in {"y", "true", "1"}: return "yes"
    # This made '1' ratings appear as 'yes' responses
    
    # Only match explicit synonyms (no digits)
    if t in {"y", "true"}:
        return "yes"
    if t in {"n", "false"}:
        return "no"
    
    return None

def open_jsonl(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, "a", encoding="utf-8")

def log_eval_row(fp, row: Dict[str, Any]) -> None:
    row = dict(row)
    row["ts"] = datetime.utcnow().isoformat() + "Z"
    fp.write(json.dumps(row, ensure_ascii=False) + "\n")

def plot_training_metrics(
    state_file: str,
    output_path: str = "training_metrics.png",
    task_id: Optional[int] = None
) -> None:
    """
    Plot training metrics from trainer_state.json.
    
    Args:
        state_file: Path to trainer_state.json
        output_path: Where to save the plot
        task_id: Optional task ID for title
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️ matplotlib not installed. Run: pip install matplotlib")
        return
    
    # Load training state
    with open(state_file, 'r') as f:
        state = json.load(f)
    
    log_history = state.get('log_history', [])
    if not log_history:
        print("⚠️ No log_history found in trainer_state.json")
        return
    
    # Extract metrics
    epochs = [log["epoch"] for log in log_history if "eval_loss" in log]
    mae = [log["eval_mae"] for log in log_history if "eval_mae" in log]
    rmse = [log.get("eval_rmse") for log in log_history if "eval_rmse" in log]
    acc = [log.get("eval_acc", log.get("eval_rating_acc")) 
           for log in log_history if "eval_acc" in log or "eval_rating_acc" in log]
    loss = [log["eval_loss"] for log in log_history if "eval_loss" in log]
    
    # Determine layout based on available metrics
    n_plots = sum([bool(mae), bool(rmse), bool(acc), bool(loss)])
    if n_plots == 0:
        print("⚠️ No evaluation metrics found")
        return
    
    # Create figure
    fig_width = min(16, n_plots * 5)
    fig, axes = plt.subplots(1, n_plots, figsize=(fig_width, 4))
    if n_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    # Plot MAE
    if mae:
        axes[plot_idx].plot(epochs, mae, marker='o', color='#e74c3c', linewidth=2)
        axes[plot_idx].set_xlabel("Epoch", fontsize=11)
        axes[plot_idx].set_ylabel("MAE", fontsize=11)
        axes[plot_idx].set_title("Mean Absolute Error", fontsize=12, fontweight='bold')
        axes[plot_idx].grid(True, alpha=0.3)
        
        # Add value annotations on last point
        if len(mae) > 0:
            axes[plot_idx].annotate(
                f'{mae[-1]:.3f}',
                xy=(epochs[-1], mae[-1]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=9,
                color='#e74c3c',
                fontweight='bold'
            )
        plot_idx += 1
    
    # Plot RMSE
    if rmse:
        axes[plot_idx].plot(epochs, rmse, marker='s', color='#9b59b6', linewidth=2)
        axes[plot_idx].set_xlabel("Epoch", fontsize=11)
        axes[plot_idx].set_ylabel("RMSE", fontsize=11)
        axes[plot_idx].set_title("Root Mean Squared Error", fontsize=12, fontweight='bold')
        axes[plot_idx].grid(True, alpha=0.3)
        
        if len(rmse) > 0:
            axes[plot_idx].annotate(
                f'{rmse[-1]:.3f}',
                xy=(epochs[-1], rmse[-1]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=9,
                color='#9b59b6',
                fontweight='bold'
            )
        plot_idx += 1
    
    # Plot Accuracy
    if acc:
        axes[plot_idx].plot(epochs, [a * 100 for a in acc], marker='o', 
                           color='#27ae60', linewidth=2)
        axes[plot_idx].set_xlabel("Epoch", fontsize=11)
        axes[plot_idx].set_ylabel("Accuracy (%)", fontsize=11)
        axes[plot_idx].set_title("Rating Accuracy", fontsize=12, fontweight='bold')
        axes[plot_idx].grid(True, alpha=0.3)
        
        if len(acc) > 0:
            axes[plot_idx].annotate(
                f'{acc[-1]*100:.1f}%',
                xy=(epochs[-1], acc[-1]*100),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=9,
                color='#27ae60',
                fontweight='bold'
            )
        plot_idx += 1
    
    # Plot Loss
    if loss:
        axes[plot_idx].plot(epochs, loss, marker='D', color='#3498db', linewidth=2)
        axes[plot_idx].set_xlabel("Epoch", fontsize=11)
        axes[plot_idx].set_ylabel("Loss", fontsize=11)
        axes[plot_idx].set_title("Evaluation Loss", fontsize=12, fontweight='bold')
        axes[plot_idx].grid(True, alpha=0.3)
        
        if len(loss) > 0:
            axes[plot_idx].annotate(
                f'{loss[-1]:.3f}',
                xy=(epochs[-1], loss[-1]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=9,
                color='#3498db',
                fontweight='bold'
            )
    
    # Add overall title
    title = "Training Metrics"
    if task_id is not None:
        title += f" - Task {task_id}"
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved metrics plot to: {output_path}")
    
    # Print summary stats
    print("\n📊 Training Summary:")
    print(f"   Epochs completed: {max(epochs) if epochs else 0:.1f}")
    if mae:
        print(f"   Final MAE: {mae[-1]:.3f} (best: {min(mae):.3f})")
    if rmse:
        print(f"   Final RMSE: {rmse[-1]:.3f} (best: {min(rmse):.3f})")
    if acc:
        print(f"   Final Accuracy: {acc[-1]*100:.1f}% (best: {max(acc)*100:.1f}%)")
    if loss:
        print(f"   Final Loss: {loss[-1]:.3f} (best: {min(loss):.3f})")