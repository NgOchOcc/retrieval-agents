"""
Evaluation metrics for multi-hop retrieval.
Implements pass@k (strict) and hit@k (relaxed) metrics.
"""

from typing import List, Dict, Set
import numpy as np
from collections import defaultdict


def pass_at_k(gold_titles: List[str], retrieved_titles: List[str], k: int) -> float:
    """
    STRICT multi-hop metric: Both supporting documents must be in top-k.

    Args:
        gold_titles: List of gold supporting document titles
        retrieved_titles: List of retrieved document titles (ordered by rank)
        k: Number of top documents to consider

    Returns:
        1.0 if all gold titles are in top-k, 0.0 otherwise
    """
    gold_set = set(gold_titles)
    retrieved_set = set(retrieved_titles[:k])

    # Check if ALL gold titles are retrieved
    if gold_set.issubset(retrieved_set):
        return 1.0
    return 0.0


def hit_at_k(gold_titles: List[str], retrieved_titles: List[str], k: int) -> float:
    """
    RELAXED metric: At least one supporting document must be in top-k.

    Args:
        gold_titles: List of gold supporting document titles
        retrieved_titles: List of retrieved document titles (ordered by rank)
        k: Number of top documents to consider

    Returns:
        1.0 if at least one gold title is in top-k, 0.0 otherwise
    """
    gold_set = set(gold_titles)
    retrieved_set = set(retrieved_titles[:k])

    # Check if ANY gold title is retrieved
    if gold_set.intersection(retrieved_set):
        return 1.0
    return 0.0


class RetrievalEvaluator:
    """Evaluates retrieval performance with pass@k and hit@k metrics."""

    def __init__(self, k_values: List[int] = [1, 5, 10, 20]):
        """
        Args:
            k_values: List of k values to evaluate
        """
        self.k_values = k_values

    def evaluate(
        self,
        gold_titles_list: List[List[str]],
        retrieved_titles_list: List[List[str]]
    ) -> Dict[str, float]:
        """
        Evaluate retrieval results across all examples.

        Args:
            gold_titles_list: List of gold title lists for each example
            retrieved_titles_list: List of retrieved title lists for each example

        Returns:
            Dictionary of metric_name -> average_score
        """
        if len(gold_titles_list) != len(retrieved_titles_list):
            raise ValueError("Mismatch between gold and retrieved lists")

        num_examples = len(gold_titles_list)
        results = defaultdict(list)

        # Compute metrics for each example
        for gold_titles, retrieved_titles in zip(gold_titles_list, retrieved_titles_list):
            for k in self.k_values:
                # pass@k (strict)
                pass_score = pass_at_k(gold_titles, retrieved_titles, k)
                results[f'pass@{k}'].append(pass_score)

                # hit@k (relaxed)
                hit_score = hit_at_k(gold_titles, retrieved_titles, k)
                results[f'hit@{k}'].append(hit_score)

        # Average across examples
        avg_results = {
            metric: np.mean(scores) for metric, scores in results.items()
        }

        return avg_results

    def evaluate_single(
        self,
        gold_titles: List[str],
        retrieved_titles: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate a single example.

        Args:
            gold_titles: Gold supporting document titles
            retrieved_titles: Retrieved document titles (ordered by rank)

        Returns:
            Dictionary of metric_name -> score
        """
        results = {}
        for k in self.k_values:
            results[f'pass@{k}'] = pass_at_k(gold_titles, retrieved_titles, k)
            results[f'hit@{k}'] = hit_at_k(gold_titles, retrieved_titles, k)
        return results

    def format_results(self, results: Dict[str, float]) -> str:
        """
        Format results as a readable table.

        Args:
            results: Dictionary of metric -> score

        Returns:
            Formatted string table
        """
        lines = ["=" * 40]
        lines.append("Retrieval Evaluation Results")
        lines.append("=" * 40)

        # Group by metric type
        pass_metrics = {k: v for k, v in results.items() if k.startswith('pass@')}
        hit_metrics = {k: v for k, v in results.items() if k.startswith('hit@')}

        # pass@k results
        lines.append("\nSTRICT (pass@k) - Both docs in top-k:")
        lines.append("-" * 40)
        for k in self.k_values:
            metric = f'pass@{k}'
            if metric in pass_metrics:
                lines.append(f"  {metric:12s}: {pass_metrics[metric]:.4f}")

        # hit@k results
        lines.append("\nRELAXED (hit@k) - At least one doc in top-k:")
        lines.append("-" * 40)
        for k in self.k_values:
            metric = f'hit@{k}'
            if metric in hit_metrics:
                lines.append(f"  {metric:12s}: {hit_metrics[metric]:.4f}")

        lines.append("=" * 40)

        return "\n".join(lines)
