
import json
import glob
import numpy as np
from scipy import stats
from pathlib import Path

def load_variant_metrics(metrics_dir, variant_name, metric_key):
    """Load all runs for a variant and extract specific metric"""
    values = []
    pattern = f"{metrics_dir}/{variant_name}_seed*.json"
    
    for mf in glob.glob(pattern):
        with open(mf, 'r') as f:
            data = json.load(f)
            if metric_key in data and data[metric_key] is not None:
                values.append(data[metric_key])
    
    return np.array(values)

def paired_ttest(baseline_values, treatment_values, metric_name):
    """Perform paired t-test"""
    if len(baseline_values) != len(treatment_values):
        print(f"⚠️  Warning: Unequal sample sizes for {metric_name}")
        return None
    
    t_stat, p_value = stats.ttest_rel(baseline_values, treatment_values)
    
    # Effect size (Cohen's d)
    diff = treatment_values - baseline_values
    cohens_d = np.mean(diff) / np.std(diff, ddof=1)
    
    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": float(cohens_d),
        "significant": p_value < 0.01,  # p < 0.01 threshold
        "baseline_mean": float(np.mean(baseline_values)),
        "treatment_mean": float(np.mean(treatment_values)),
        "improvement": float(np.mean(treatment_values) - np.mean(baseline_values))
    }

def main():
    metrics_dir = "./ablation_results/metrics"
    
    print("🔬 Computing Statistical Significance")
    print("=" * 60)
    print("")
    
    # Metrics to compare (lower is better for MAE/RMSE, higher for accuracy)
    comparisons = [
        ("baseline", "full", "mae", "lower_better"),
        ("baseline", "full", "rmse", "lower_better"),
        ("baseline", "full", "accuracy", "higher_better"),
    ]
    
    results = {}
    
    for baseline_var, treatment_var, metric, direction in comparisons:
        print(f"📊 Comparing {baseline_var} vs {treatment_var} on {metric}:")
        
        baseline_vals = load_variant_metrics(metrics_dir, baseline_var, metric)
        treatment_vals = load_variant_metrics(metrics_dir, treatment_var, metric)
        
        if len(baseline_vals) == 0 or len(treatment_vals) == 0:
            print(f"   ⚠️  Insufficient data\n")
            continue
        
        # For "lower is better" metrics, we want treatment < baseline
        # For "higher is better", we want treatment > baseline
        if direction == "lower_better":
            # Negate so improvement is positive
            test_result = paired_ttest(baseline_vals, -treatment_vals, metric)
        else:
            test_result = paired_ttest(baseline_vals, treatment_vals, metric)
        
        if test_result:
            results[f"{baseline_var}_vs_{treatment_var}_{metric}"] = test_result
            
            print(f"   Baseline: {test_result['baseline_mean']:.4f}")
            print(f"   Treatment: {test_result['treatment_mean']:.4f}")
            print(f"   Improvement: {abs(test_result['improvement']):.4f}")
            print(f"   p-value: {test_result['p_value']:.4f}")
            print(f"   Cohen's d: {test_result['cohens_d']:.3f}")
            
            if test_result['significant']:
                print(f"   ✅ Statistically significant (p < 0.01)")
            else:
                print(f"   ❌ Not significant (p >= 0.01)")
            print("")
    
    # Save results
    output_file = "./ablation_results/significance_tests.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Saved significance tests to {output_file}")

if __name__ == "__main__":
    main()