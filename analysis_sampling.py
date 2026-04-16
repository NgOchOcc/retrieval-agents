#!/usr/bin/env python3
"""
Analyze sampling experiment results to find optimal parameters.
Compares baseline vs sampling across different p, n, k values.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np


def load_result_file(filepath: Path) -> dict:
    """Load a single result JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_experiment_params(result: dict) -> tuple:
    """Extract (n, p, k) from result."""
    params = result['sampling_params']
    return params['n'], params['p'], params['k']


def compare_baseline_vs_sampling(result: dict) -> dict:
    """Compare baseline and sampling metrics."""
    baseline = result['results']['baseline']
    sampling = result['results']['sampling']

    comparison = {}
    for metric in baseline.keys():
        baseline_val = baseline[metric]
        sampling_val = sampling[metric]
        diff = sampling_val - baseline_val
        improvement = (diff / baseline_val * 100) if baseline_val > 0 else 0

        comparison[metric] = {
            'baseline': baseline_val,
            'sampling': sampling_val,
            'diff': diff,
            'improvement_pct': improvement
        }

    return comparison


def analyze_directory(results_dir: Path, output_file: Path = None):
    """Analyze all result files in directory."""
    result_files = list(results_dir.glob("sampling_comparison_*.json"))

    if not result_files:
        print(f"No result files found in {results_dir}")
        return

    print(f"Found {len(result_files)} result files")
    print("=" * 80)

    # Group results by parameters
    results_by_p = defaultdict(list)
    results_by_n = defaultdict(list)
    all_comparisons = []

    for filepath in result_files:
        result = load_result_file(filepath)
        n, p, k = extract_experiment_params(result)
        comparison = compare_baseline_vs_sampling(result)

        results_by_p[p].append((n, k, comparison))
        results_by_n[n].append((p, k, comparison))
        all_comparisons.append({
            'n': n,
            'p': p,
            'k': k,
            'comparison': comparison,
            'file': filepath.name
        })

    # Find best configurations
    print("\n📊 BEST CONFIGURATIONS")
    print("=" * 80)

    metrics_to_check = ['pass@5', 'pass@10', 'pass@20', 'hit@5', 'hit@10', 'hit@20']

    for metric in metrics_to_check:
        best_config = None
        best_improvement = -float('inf')

        for config in all_comparisons:
            if metric in config['comparison']:
                improvement = config['comparison'][metric]['diff']
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_config = config

        if best_config:
            comp = best_config['comparison'][metric]
            print(f"\n{metric}:")
            print(f"  Best config: n={best_config['n']}, p={best_config['p']:.2f}, k={best_config['k']}")
            print(f"  Baseline:    {comp['baseline']:.4f}")
            print(f"  Sampling:    {comp['sampling']:.4f}")
            print(f"  Difference:  {comp['diff']:+.4f} ({comp['improvement_pct']:+.2f}%)")
            if comp['diff'] > 0:
                print(f"  ✅ Sampling WINS!")
            else:
                print(f"  ❌ Baseline better")

    # Analyze by p value
    print("\n\n📈 ANALYSIS BY SAMPLING PROBABILITY (p)")
    print("=" * 80)

    p_values = sorted(results_by_p.keys())
    for p in p_values:
        configs = results_by_p[p]
        print(f"\np = {p:.2f} ({len(configs)} experiments):")

        # Average improvements across all configs with this p
        avg_improvements = defaultdict(list)
        for n, k, comparison in configs:
            for metric, values in comparison.items():
                avg_improvements[metric].append(values['diff'])

        # Show average improvement for key metrics
        for metric in ['pass@5', 'pass@10', 'pass@20']:
            if metric in avg_improvements:
                avg_diff = np.mean(avg_improvements[metric])
                print(f"  {metric:12s}: {avg_diff:+.4f} avg difference")

    # Analyze by n value
    print("\n\n📈 ANALYSIS BY TOP-N RETRIEVED")
    print("=" * 80)

    n_values = sorted(results_by_n.keys())
    for n in n_values:
        configs = results_by_n[n]
        print(f"\nn = {n} ({len(configs)} experiments):")

        avg_improvements = defaultdict(list)
        for p, k, comparison in configs:
            for metric, values in comparison.items():
                avg_improvements[metric].append(values['diff'])

        for metric in ['pass@5', 'pass@10', 'pass@20']:
            if metric in avg_improvements:
                avg_diff = np.mean(avg_improvements[metric])
                print(f"  {metric:12s}: {avg_diff:+.4f} avg difference")

    # Summary statistics
    print("\n\n📊 SUMMARY STATISTICS")
    print("=" * 80)

    for metric in metrics_to_check:
        improvements = []
        for config in all_comparisons:
            if metric in config['comparison']:
                improvements.append(config['comparison'][metric]['diff'])

        if improvements:
            improvements = np.array(improvements)
            print(f"\n{metric}:")
            print(f"  Avg difference:    {np.mean(improvements):+.4f}")
            print(f"  Std deviation:     {np.std(improvements):.4f}")
            print(f"  Min difference:    {np.min(improvements):+.4f}")
            print(f"  Max difference:    {np.max(improvements):+.4f}")
            print(f"  Wins (sampling better): {np.sum(improvements > 0)}/{len(improvements)}")

    # Save detailed report if requested
    if output_file:
        report = {
            'summary': {
                'total_experiments': len(all_comparisons),
                'p_values_tested': sorted(results_by_p.keys()),
                'n_values_tested': sorted(results_by_n.keys()),
            },
            'all_comparisons': all_comparisons
        }

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n\n💾 Detailed report saved to: {output_file}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze sampling experiment results"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/sampling",
        help="Directory containing result JSON files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for detailed report (optional)"
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_file = Path(args.output) if args.output else None

    analyze_directory(results_dir, output_file)


if __name__ == "__main__":
    main()
