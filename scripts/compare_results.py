"""
Results Comparison Utility for LoL Match Prediction System.

Compares training runs against baseline and tracks improvements.
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

RESULTS_DIR = project_root / "outputs" / "results"
BASELINE_FILE = RESULTS_DIR / "baseline_results.json"
HISTORY_FILE = RESULTS_DIR / "training_history.json"


def ensure_results_dir():
    """Ensure results directory exists."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def save_baseline(results: Dict, description: str = ""):
    """Save current results as the baseline.

    Args:
        results: Dictionary of results from training
        description: Optional description of this baseline
    """
    ensure_results_dir()

    baseline = {
        'timestamp': datetime.now().isoformat(),
        'description': description or "Baseline results",
        'results': results
    }

    with open(BASELINE_FILE, 'w') as f:
        json.dump(baseline, f, indent=2, default=str)

    print(f"Baseline saved to: {BASELINE_FILE}")


def load_baseline() -> Optional[Dict]:
    """Load baseline results."""
    if not BASELINE_FILE.exists():
        return None

    with open(BASELINE_FILE, 'r') as f:
        return json.load(f)


def save_run(results: Dict, description: str = ""):
    """Save a training run to history.

    Args:
        results: Dictionary of results from training
        description: Optional description of this run
    """
    ensure_results_dir()

    # Load existing history
    history = load_history()

    # Add new run
    run = {
        'timestamp': datetime.now().isoformat(),
        'description': description,
        'results': results
    }
    history.append(run)

    # Save updated history
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2, default=str)

    print(f"Run saved to history ({len(history)} total runs)")
    return len(history) - 1  # Return run index


def load_history() -> List[Dict]:
    """Load training history."""
    if not HISTORY_FILE.exists():
        return []

    with open(HISTORY_FILE, 'r') as f:
        return json.load(f)


def compare_to_baseline(current_results: Dict) -> Dict:
    """Compare current results to baseline.

    Args:
        current_results: Current training results

    Returns:
        Dictionary with comparison metrics
    """
    baseline = load_baseline()

    if baseline is None:
        print("No baseline found. Save one with: compare_results.save_baseline(results)")
        return {'error': 'No baseline found'}

    comparison = {
        'baseline_timestamp': baseline['timestamp'],
        'baseline_description': baseline['description'],
        'current_timestamp': datetime.now().isoformat(),
        'metrics': {}
    }

    baseline_results = baseline['results']

    # Key metrics to compare
    key_metrics = [
        'test_auc', 'test_f1', 'test_accuracy', 'test_mcc', 'test_kappa',
        'test_log_loss', 'test_brier', 'test_ece', 'generalization_gap'
    ]

    for metric in key_metrics:
        if metric in current_results and metric in baseline_results:
            current = current_results[metric]
            baseline_val = baseline_results[metric]

            if isinstance(current, (int, float)) and isinstance(baseline_val, (int, float)):
                # Higher is better for most metrics, except loss/gap
                lower_is_better = metric in ['test_log_loss', 'test_brier', 'test_ece', 'generalization_gap']

                if lower_is_better:
                    improvement = baseline_val - current
                    is_better = current < baseline_val
                else:
                    improvement = current - baseline_val
                    is_better = current > baseline_val

                comparison['metrics'][metric] = {
                    'baseline': baseline_val,
                    'current': current,
                    'improvement': improvement,
                    'improvement_pct': (improvement / abs(baseline_val) * 100) if baseline_val != 0 else 0,
                    'is_better': is_better
                }

    return comparison


def print_comparison(comparison: Dict):
    """Print formatted comparison results."""
    if 'error' in comparison:
        print(f"\nError: {comparison['error']}")
        return

    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)

    print(f"\nBaseline: {comparison['baseline_description']}")
    print(f"  Timestamp: {comparison['baseline_timestamp']}")
    print(f"\nCurrent Run: {comparison['current_timestamp']}")

    print("\n" + "-" * 70)
    print(f"{'Metric':<25} {'Baseline':>10} {'Current':>10} {'Change':>10} {'Status':>10}")
    print("-" * 70)

    for metric, values in comparison['metrics'].items():
        baseline = values['baseline']
        current = values['current']
        improvement = values['improvement']
        is_better = values['is_better']

        status = "BETTER" if is_better else "WORSE" if not is_better else "SAME"
        sign = "+" if improvement >= 0 else ""

        print(f"{metric:<25} {baseline:>10.4f} {current:>10.4f} {sign}{improvement:>9.4f} {status:>10}")

    print("-" * 70)

    # Summary
    better_count = sum(1 for v in comparison['metrics'].values() if v['is_better'])
    worse_count = sum(1 for v in comparison['metrics'].values() if not v['is_better'])

    print(f"\nSummary: {better_count} metrics improved, {worse_count} metrics declined")


def get_best_run() -> Optional[Dict]:
    """Get the best run from history based on composite score."""
    history = load_history()

    if not history:
        return None

    best_run = None
    best_score = -1

    for run in history:
        results = run.get('results', {})
        # Calculate simple composite score
        auc = results.get('test_auc', 0)
        f1 = results.get('test_f1', 0)
        mcc = results.get('test_mcc', 0)

        score = (auc + f1 + (mcc + 1) / 2) / 3  # Average of key metrics

        if score > best_score:
            best_score = score
            best_run = run

    return best_run


def list_runs(n: int = 10):
    """List recent training runs.

    Args:
        n: Number of recent runs to show
    """
    history = load_history()

    if not history:
        print("No training runs recorded yet.")
        return

    print("\n" + "=" * 70)
    print(f"TRAINING HISTORY (last {n} runs)")
    print("=" * 70)

    for i, run in enumerate(history[-n:]):
        results = run.get('results', {})
        print(f"\nRun #{len(history) - n + i + 1}")
        print(f"  Timestamp: {run['timestamp']}")
        print(f"  Description: {run.get('description', 'N/A')}")
        print(f"  Model: {results.get('model', 'N/A')}")
        print(f"  AUC: {results.get('test_auc', 'N/A'):.4f}" if 'test_auc' in results else "  AUC: N/A")
        print(f"  F1: {results.get('test_f1', 'N/A'):.4f}" if 'test_f1' in results else "  F1: N/A")
        print(f"  MCC: {results.get('test_mcc', 'N/A'):.4f}" if 'test_mcc' in results else "  MCC: N/A")


def create_baseline_from_thesis():
    """Create baseline from the thesis results (82.97% AUC)."""
    baseline_results = {
        'model': 'Logistic Regression',
        'test_auc': 0.8297,
        'test_f1': 0.7650,  # Estimated from thesis
        'test_accuracy': 0.7550,  # Estimated
        'test_mcc': 0.5100,  # Estimated
        'test_kappa': 0.5100,  # Estimated
        'test_log_loss': 0.4800,  # Estimated
        'test_brier': 0.1800,  # Estimated
        'test_ece': 0.0500,  # Estimated
        'generalization_gap': 0.0100
    }

    save_baseline(baseline_results, "Thesis baseline (Logistic Regression, 82.97% AUC)")
    print("\nBaseline created from thesis results.")
    print("Key metric: AUC-ROC = 82.97%")


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="Compare training results")
    parser.add_argument('--create-thesis-baseline', action='store_true',
                        help='Create baseline from thesis results')
    parser.add_argument('--list', action='store_true', help='List training history')
    parser.add_argument('--best', action='store_true', help='Show best run')
    parser.add_argument('-n', type=int, default=10, help='Number of runs to show')

    args = parser.parse_args()

    if args.create_thesis_baseline:
        create_baseline_from_thesis()
    elif args.list:
        list_runs(args.n)
    elif args.best:
        best = get_best_run()
        if best:
            print("\nBest Run:")
            print(f"  Timestamp: {best['timestamp']}")
            print(f"  Results: {json.dumps(best['results'], indent=4)}")
        else:
            print("No runs recorded yet.")
    else:
        # Show current baseline
        baseline = load_baseline()
        if baseline:
            print("\nCurrent Baseline:")
            print(f"  Description: {baseline['description']}")
            print(f"  Timestamp: {baseline['timestamp']}")
            print(f"  Key Results:")
            results = baseline['results']
            for key in ['test_auc', 'test_f1', 'test_mcc']:
                if key in results:
                    print(f"    {key}: {results[key]:.4f}")
        else:
            print("No baseline set. Create one with --create-thesis-baseline")


if __name__ == "__main__":
    main()
