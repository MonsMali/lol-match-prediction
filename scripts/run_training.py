"""
Training Runner Script for LoL Match Prediction System.

Runs model training with all enhancements and saves results for comparison.
"""

import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.trainer import main as train_main
from scripts.compare_results import save_run, compare_to_baseline, print_comparison


def run_training(
    quick_mode: bool = False,
    leakage_free: bool = True,
    split_method: str = 'stratified_temporal',
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    use_temporal_weighting: bool = False,
    calibrate_probs: bool = True,
    quantify_uncertainty: bool = True,
    description: str = "",
    # Legacy flags (only used when leakage_free=False)
    use_stratified_temporal: bool = True,
    use_enhanced_v2: bool = True,
):
    """Run training with specified configuration.

    Args:
        quick_mode: Use reduced hyperparameter search
        leakage_free: Use leakage-free pipeline (split first, then compute stats)
        split_method: 'stratified_temporal' or 'pure_temporal' (leakage_free only)
        train_size: Training set proportion (leakage_free only)
        val_size: Validation set proportion (leakage_free only)
        test_size: Test set proportion (leakage_free only)
        use_temporal_weighting: Apply temporal sample weighting
        calibrate_probs: Apply probability calibration
        quantify_uncertainty: Compute uncertainty quantification
        description: Description for this training run
        use_stratified_temporal: Legacy flag (leakage_free=False only)
        use_enhanced_v2: Legacy flag (leakage_free=False only)
    """
    print("=" * 70)
    print("LOL MATCH PREDICTION - TRAINING RUN")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"\nConfiguration:")
    print(f"  Quick mode: {quick_mode}")
    print(f"  Leakage-free pipeline: {leakage_free}")
    if leakage_free:
        print(f"  Split method: {split_method}")
        print(f"  Split ratios: {train_size:.0%}/{val_size:.0%}/{test_size:.0%}")
    else:
        print(f"  Stratified temporal: {use_stratified_temporal}")
        print(f"  Enhanced v2 features: {use_enhanced_v2}")
    print(f"  Temporal weighting: {use_temporal_weighting}")
    print(f"  Probability calibration: {calibrate_probs}")
    print(f"  Uncertainty quantification: {quantify_uncertainty}")

    # Run training
    predictor, results = train_main(
        quick_mode=quick_mode,
        leakage_free=leakage_free,
        split_method=split_method,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        use_stratified_temporal=use_stratified_temporal,
        use_enhanced_v2=use_enhanced_v2,
        use_temporal_weighting=use_temporal_weighting,
        calibrate_probs=calibrate_probs,
        quantify_uncertainty=quantify_uncertainty
    )

    # Prepare results for saving
    save_results = {
        'model': results.get('model', 'Unknown'),
        'test_auc': results.get('test_auc'),
        'test_f1': results.get('test_f1'),
        'test_accuracy': results.get('test_accuracy'),
        'test_mcc': results.get('test_mcc'),
        'test_kappa': results.get('test_kappa'),
        'test_balanced_accuracy': results.get('test_balanced_accuracy'),
        'test_log_loss': results.get('test_log_loss'),
        'test_brier': results.get('test_brier'),
        'test_ece': results.get('test_ece'),
        'generalization_gap': results.get('generalization_gap'),
        'config': {
            'quick_mode': quick_mode,
            'leakage_free': leakage_free,
            'split_method': split_method if leakage_free else 'legacy',
            'use_temporal_weighting': use_temporal_weighting,
            'calibrate_probs': calibrate_probs,
            'quantify_uncertainty': quantify_uncertainty
        }
    }

    # Add calibration results if available
    if results.get('calibration_result'):
        cal = results['calibration_result']
        save_results['calibration'] = {
            'method': cal.method,
            'ece_before': cal.ece_before,
            'ece_after': cal.ece_after,
            'improvement': cal.improvement
        }

    # Save to history
    if not description:
        description = f"Training run - {'Leakage-free' if leakage_free else 'Legacy'} pipeline"
    save_run(save_results, description)

    # Compare to baseline
    print("\n" + "=" * 70)
    print("COMPARISON TO BASELINE")
    print("=" * 70)

    comparison = compare_to_baseline(save_results)
    print_comparison(comparison)

    return predictor, results, comparison


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="Run LoL match prediction training")
    parser.add_argument('--quick', action='store_true', help='Quick mode (reduced search)')
    parser.add_argument('--legacy', action='store_true', help='Use legacy pipeline (leaky, for comparison)')
    parser.add_argument('--pure-temporal', action='store_true', help='Use pure temporal split')
    parser.add_argument('--temporal-weighting', action='store_true', help='Enable temporal weighting')
    parser.add_argument('--no-calibration', action='store_true', help='Disable probability calibration')
    parser.add_argument('--no-uncertainty', action='store_true', help='Disable uncertainty quantification')
    parser.add_argument('--description', type=str, default='', help='Description for this run')

    args = parser.parse_args()

    run_training(
        quick_mode=args.quick,
        leakage_free=not args.legacy,
        split_method='pure_temporal' if args.pure_temporal else 'stratified_temporal',
        use_temporal_weighting=args.temporal_weighting,
        calibrate_probs=not args.no_calibration,
        quantify_uncertainty=not args.no_uncertainty,
        description=args.description
    )


if __name__ == "__main__":
    main()
