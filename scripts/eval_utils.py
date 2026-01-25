
import re

class EvalTracker:
    """
    Tracks and logs example-by-example evaluation results,
    computes final accuracy.
    """
    def __init__(self):
        self.examples = []
        self.matches = 0
        self.total = 0

    @staticmethod
    def _clean_text(text: str) -> str:
        """Lowercases and strips common punctuation for fair comparison."""
        if text is None:
            return ""
        return re.sub(r"\s+", " ", str(text)).strip().lower()

    def log_example(self, idx: int, input_text: str, pred_text: str, ground_truth: str = None):
        clean_pred = self._clean_text(pred_text)
        clean_truth = self._clean_text(ground_truth) if ground_truth is not None else None
        match = (clean_truth is not None and clean_pred == clean_truth)
        
        # Update counters
        if match:
            self.matches += 1
        if ground_truth is not None:
            self.total += 1

        # Store example for possible later dumping
        self.examples.append({
            "idx": idx,
            "input": input_text,
            "pred": pred_text,
            "pred_clean": clean_pred,
            "truth": ground_truth,
            "truth_clean": clean_truth,
            "match": match
        })

        # Print immediately
        print(f"\n--- Example {idx} ---")
        print(f"Input: {input_text}")
        print(f"Predicted: '{pred_text}' (Cleaned: '{clean_pred}')")
        if ground_truth is not None:
            print(f"Ground Truth: '{ground_truth}' (Cleaned: '{clean_truth}')")
            print(f"Match: {match}")
        else:
            print("Ground Truth: [N/A]")

    def summarize(self):
        print("\n" + "=" * 60)
        if self.total > 0:
            accuracy = self.matches / self.total
            print(f"Evaluation complete: {self.matches}/{self.total} matches → Accuracy: {accuracy:.4f}")
        else:
            print("Evaluation complete: No ground truth labels available.")
        print("=" * 60)
