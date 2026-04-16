#!/usr/bin/env python3
"""Script to compare results from different retrieval settings."""

import json
import glob
import os
from datetime import datetime
from typing import Dict, List
import argparse


def load_result_file(filepath: str) -> Dict:
    """Load a result JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def format_metric(value: float, is_percentage: bool = True) -> str:
    """Format metric value."""
    if is_percentage:
        return f"{value * 100:.2f}%"
    else:
        return f"{value:.4f}"


def print_comparison_table(results: List[Dict]):
    """Print a comparison table of results."""
    if not results:
        print("No results to compare.")
        return

    print("\n" + "=" * 120)
    print("RETRIEVAL BENCHMARK COMPARISON")
    print("=" * 120)

    # Group by model
    models = {}
    for result in results:
        model_name = result['model_name']
        if model_name not in models:
            models[model_name] = []
        models[model_name].append(result)

    for model_name, model_results in models.items():
        print(f"\nModel: {model_name}")
        print("-" * 120)

        # Header
        header = f"{'Strategy':<20} | "
        metrics_keys = ['recall@1', 'recall@5', 'recall@10', 'recall@20', 'mrr', 'precision@5']

        for metric in metrics_keys:
            if any(metric in r['metrics'] for r in model_results):
                header += f"{metric.upper():<12} | "

        print(header)
        print("-" * 120)

        # Rows
        for result in model_results:
            strategy = result['sampling_strategy']
            if strategy == 'top_k':
                strategy_display = "Normal (Top-K)"
            elif strategy == 'random_sample':
                exp_factor = result.get('expansion_factor', 'N/A')
                rand_ratio = result.get('random_ratio', 'N/A')
                strategy_display = f"Random (×{exp_factor}, r={rand_ratio})"
            elif strategy == 'diverse_sample':
                exp_factor = result.get('expansion_factor', 'N/A')
                rand_ratio = result.get('random_ratio', 'N/A')
                strategy_display = f"Diverse (×{exp_factor}, r={rand_ratio})"
            else:
                strategy_display = strategy

            row = f"{strategy_display:<20} | "

            for metric in metrics_keys:
                if metric in result['metrics']:
                    value = result['metrics'][metric]
                    is_pct = metric.startswith('recall') or metric.startswith('precision')
                    row += f"{format_metric(value, is_pct):<12} | "

            print(row)

        print("-" * 120)

    print("\n" + "=" * 120)


def print_detailed_comparison(results: List[Dict]):
    """Print detailed comparison with all k values."""
    if not results:
        return

    print("\n" + "=" * 120)
    print("DETAILED METRICS COMPARISON (All K Values)")
    print("=" * 120)

    strategies = {}
    for result in results:
        strategy = result['sampling_strategy']
        if strategy not in strategies:
            strategies[strategy] = result

    # Print Recall@K for all k values
    print("\n📊 Recall@K Comparison:")
    print("-" * 80)
    header = f"{'Strategy':<20} | "
    k_values = [1, 3, 5, 10, 20, 50, 100]

    for k in k_values:
        metric_key = f'recall@{k}'
        if any(metric_key in r['metrics'] for r in strategies.values()):
            header += f"R@{k:<3} | "

    print(header)
    print("-" * 80)

    for strategy, result in strategies.items():
        if strategy == 'top_k':
            strategy_display = "Normal (Top-K)"
        elif strategy == 'random_sample':
            strategy_display = "Random Sample"
        elif strategy == 'diverse_sample':
            strategy_display = "Diverse Sample"
        else:
            strategy_display = strategy

        row = f"{strategy_display:<20} | "

        for k in k_values:
            metric_key = f'recall@{k}'
            if metric_key in result['metrics']:
                value = result['metrics'][metric_key]
                row += f"{value*100:>5.2f}% | "

        print(row)

    print("-" * 80)

    # Print Precision@K
    print("\n🎯 Precision@K Comparison:")
    print("-" * 80)
    print(header.replace('R@', 'P@'))
    print("-" * 80)

    for strategy, result in strategies.items():
        if strategy == 'top_k':
            strategy_display = "Normal (Top-K)"
        elif strategy == 'random_sample':
            strategy_display = "Random Sample"
        elif strategy == 'diverse_sample':
            strategy_display = "Diverse Sample"
        else:
            strategy_display = strategy

        row = f"{strategy_display:<20} | "

        for k in k_values:
            metric_key = f'precision@{k}'
            if metric_key in result['metrics']:
                value = result['metrics'][metric_key]
                row += f"{value*100:>5.2f}% | "

        print(row)

    print("-" * 80)
    print("\n" + "=" * 120)


def print_best_strategy(results: List[Dict]):
    """Print which strategy performed best for each metric."""
    if not results:
        return

    print("\n" + "=" * 120)
    print("🏆 BEST PERFORMING STRATEGY FOR EACH METRIC")
    print("=" * 120)

    # Collect all metrics
    all_metrics = set()
    for result in results:
        all_metrics.update(result['metrics'].keys())

    # Find best for each metric
    for metric in sorted(all_metrics):
        best_value = -1
        best_strategy = None

        for result in results:
            if metric in result['metrics']:
                value = result['metrics'][metric]
                if value > best_value:
                    best_value = value
                    best_strategy = result['sampling_strategy']

        if best_strategy:
            strategy_display = {
                'top_k': 'Normal (Top-K)',
                'random_sample': 'Random Sample',
                'diverse_sample': 'Diverse Sample'
            }.get(best_strategy, best_strategy)

            print(f"{metric.upper():<20}: {strategy_display:<20} ({format_metric(best_value)})")

    print("=" * 120)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Compare retrieval benchmark results")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./cache/results",
        help="Directory containing result JSON files"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.json",
        help="Pattern to match result files"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed comparison with all k values"
    )

    args = parser.parse_args()

    # Find result files
    pattern = os.path.join(args.results_dir, args.pattern)
    result_files = glob.glob(pattern)

    if not result_files:
        print(f"No result files found matching: {pattern}")
        return

    # Load results
    results = []
    for filepath in sorted(result_files):
        try:
            result = load_result_file(filepath)
            results.append(result)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")

    if not results:
        print("No valid results loaded.")
        return

    print(f"\nLoaded {len(results)} result files from: {args.results_dir}")

    # Print comparison table
    print_comparison_table(results)

    # Print detailed comparison if requested
    if args.detailed:
        print_detailed_comparison(results)

    # Print best strategy summary
    print_best_strategy(results)

    print("\n")
    print("💡 Tip: Run with --detailed flag to see all k values")
    print("💡 Tip: Run with --pattern to filter specific results")
    print("")


if __name__ == "__main__":
    main()
