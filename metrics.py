"""Evaluation metrics for retrieval."""

import numpy as np
from typing import List, Dict, Set
from collections import defaultdict


class RetrievalMetrics:
    """Calculate retrieval evaluation metrics."""

    @staticmethod
    def pass_at_k(retrieved_docs: List[str], relevant_docs: Set[str], k: int) -> float:
        """
        Calculate Pass@k metric (also known as Recall@k or Success@k).

        Pass@k = 1 if at least one relevant document is in top-k, 0 otherwise

        Args:
            retrieved_docs: List of retrieved document IDs (ranked)
            relevant_docs: Set of relevant document IDs
            k: Cutoff rank

        Returns:
            1.0 if successful, 0.0 otherwise
        """
        if not relevant_docs:
            return 0.0

        top_k_docs = set(retrieved_docs[:k])
        return 1.0 if len(top_k_docs & relevant_docs) > 0 else 0.0

    @staticmethod
    def recall_at_k(retrieved_docs: List[str], relevant_docs: Set[str], k: int) -> float:
        """
        Calculate Recall@k.

        Recall@k = (# relevant docs in top-k) / (# total relevant docs)

        Args:
            retrieved_docs: List of retrieved document IDs (ranked)
            relevant_docs: Set of relevant document IDs
            k: Cutoff rank

        Returns:
            Recall@k score
        """
        if not relevant_docs:
            return 0.0

        top_k_docs = set(retrieved_docs[:k])
        num_relevant_retrieved = len(top_k_docs & relevant_docs)
        return num_relevant_retrieved / len(relevant_docs)

    @staticmethod
    def precision_at_k(retrieved_docs: List[str], relevant_docs: Set[str], k: int) -> float:
        """
        Calculate Precision@k.

        Precision@k = (# relevant docs in top-k) / k

        Args:
            retrieved_docs: List of retrieved document IDs (ranked)
            relevant_docs: Set of relevant document IDs
            k: Cutoff rank

        Returns:
            Precision@k score
        """
        if not relevant_docs or k == 0:
            return 0.0

        top_k_docs = set(retrieved_docs[:k])
        num_relevant_retrieved = len(top_k_docs & relevant_docs)
        return num_relevant_retrieved / k

    @staticmethod
    def ndcg_at_k(
        retrieved_docs: List[str],
        relevance_labels: Dict[str, int],
        k: int,
    ) -> float:
        """
        Calculate NDCG@k (Normalized Discounted Cumulative Gain).

        NDCG accounts for both relevance and ranking position.

        Args:
            retrieved_docs: List of retrieved document IDs (ranked)
            relevance_labels: Dict mapping doc_id to relevance score (0 or 1 for binary)
            k: Cutoff rank

        Returns:
            NDCG@k score
        """
        if not relevance_labels:
            return 0.0

        # Calculate DCG@k
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_docs[:k]):
            rel = relevance_labels.get(doc_id, 0)
            # DCG formula: sum(rel_i / log2(i+2))
            dcg += rel / np.log2(i + 2)

        # Calculate IDCG@k (ideal DCG)
        # Sort relevance scores in descending order
        ideal_rels = sorted(relevance_labels.values(), reverse=True)[:k]
        idcg = 0.0
        for i, rel in enumerate(ideal_rels):
            idcg += rel / np.log2(i + 2)

        # NDCG = DCG / IDCG
        if idcg == 0.0:
            return 0.0
        return dcg / idcg

    @staticmethod
    def mean_reciprocal_rank(retrieved_docs: List[str], relevant_docs: Set[str]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).

        MRR = 1 / rank of first relevant document

        Args:
            retrieved_docs: List of retrieved document IDs (ranked)
            relevant_docs: Set of relevant document IDs

        Returns:
            MRR score
        """
        if not relevant_docs:
            return 0.0

        for i, doc_id in enumerate(retrieved_docs):
            if doc_id in relevant_docs:
                return 1.0 / (i + 1)

        return 0.0


class BenchmarkEvaluator:
    """Evaluate retrieval results across multiple queries."""

    def __init__(self, k_values: List[int] = None):
        """
        Initialize evaluator.

        Args:
            k_values: List of k values for evaluation (default: [1, 3, 5, 10, 20])
        """
        self.k_values = k_values or [1, 3, 5, 10, 20]
        self.metrics_calculator = RetrievalMetrics()

    def evaluate(
        self,
        retrieval_results: Dict[str, List[str]],
        ground_truth: Dict[str, Dict[str, int]],
    ) -> Dict[str, float]:
        """
        Evaluate retrieval results.

        Args:
            retrieval_results: Dict mapping question_id -> list of retrieved doc_ids (ranked)
            ground_truth: Dict mapping question_id -> {doc_id: relevance_label}

        Returns:
            Dict of metric_name -> score
        """
        if not retrieval_results:
            print("WARNING: No retrieval results to evaluate")
            return {}

        # Initialize metric storage
        metrics = defaultdict(list)

        # Track statistics
        num_evaluated = 0
        num_skipped_no_ground_truth = 0
        num_skipped_no_relevant = 0

        # Evaluate each query
        for question_id, retrieved_docs in retrieval_results.items():
            if question_id not in ground_truth:
                num_skipped_no_ground_truth += 1
                continue

            relevance_labels = ground_truth[question_id]
            relevant_docs = set(doc_id for doc_id, label in relevance_labels.items() if label > 0)

            # Skip if no relevant docs
            if not relevant_docs:
                num_skipped_no_relevant += 1
                continue

            num_evaluated += 1

            # Calculate metrics at different k values
            for k in self.k_values:
                # Pass@k (Success@k)
                pass_k = self.metrics_calculator.pass_at_k(retrieved_docs, relevant_docs, k)
                metrics[f"pass@{k}"].append(pass_k)

                # Recall@k
                recall_k = self.metrics_calculator.recall_at_k(retrieved_docs, relevant_docs, k)
                metrics[f"recall@{k}"].append(recall_k)

                # Precision@k
                precision_k = self.metrics_calculator.precision_at_k(retrieved_docs, relevant_docs, k)
                metrics[f"precision@{k}"].append(precision_k)

                # NDCG@k
                ndcg_k = self.metrics_calculator.ndcg_at_k(retrieved_docs, relevance_labels, k)
                metrics[f"ndcg@{k}"].append(ndcg_k)

            # MRR (no k cutoff)
            mrr = self.metrics_calculator.mean_reciprocal_rank(retrieved_docs, relevant_docs)
            metrics["mrr"].append(mrr)

        # Print statistics
        print(f"\nEvaluation Statistics:")
        print(f"  Total questions: {len(retrieval_results)}")
        print(f"  Evaluated: {num_evaluated}")
        print(f"  Skipped (no ground truth): {num_skipped_no_ground_truth}")
        print(f"  Skipped (no relevant docs): {num_skipped_no_relevant}")

        if num_evaluated == 0:
            print("\nERROR: No questions were evaluated!")
            print("Possible issues:")
            print("  1. Question IDs don't match between retrieval results and ground truth")
            print("  2. No supporting facts found in the corpus")
            return {}

        # Aggregate metrics (mean across all queries)
        aggregated_metrics = {}
        for metric_name, values in metrics.items():
            aggregated_metrics[metric_name] = np.mean(values) if values else 0.0

        return aggregated_metrics

    def print_results(self, metrics: Dict[str, float], model_name: str = "Model"):
        """
        Print evaluation results in a formatted way.

        Args:
            metrics: Dict of metric_name -> score
            model_name: Name of the model being evaluated
        """
        print(f"\n{'=' * 60}")
        print(f"Evaluation Results: {model_name}")
        print(f"{'=' * 60}")

        # Group metrics by type
        pass_metrics = {k: v for k, v in metrics.items() if k.startswith("pass@")}
        recall_metrics = {k: v for k, v in metrics.items() if k.startswith("recall@")}
        precision_metrics = {k: v for k, v in metrics.items() if k.startswith("precision@")}
        ndcg_metrics = {k: v for k, v in metrics.items() if k.startswith("ndcg@")}
        other_metrics = {
            k: v for k, v in metrics.items() if not any(k.startswith(p) for p in ["pass@", "recall@", "precision@", "ndcg@"])
        }

        # Print Pass@k
        if pass_metrics:
            print("\nPass@k (Success Rate):")
            for k in sorted(self.k_values):
                metric_name = f"pass@{k}"
                if metric_name in pass_metrics:
                    print(f"  {metric_name:15s}: {pass_metrics[metric_name]:.4f}")

        # Print Recall@k
        if recall_metrics:
            print("\nRecall@k:")
            for k in sorted(self.k_values):
                metric_name = f"recall@{k}"
                if metric_name in recall_metrics:
                    print(f"  {metric_name:15s}: {recall_metrics[metric_name]:.4f}")

        # Print Precision@k
        if precision_metrics:
            print("\nPrecision@k:")
            for k in sorted(self.k_values):
                metric_name = f"precision@{k}"
                if metric_name in precision_metrics:
                    print(f"  {metric_name:15s}: {precision_metrics[metric_name]:.4f}")

        # Print NDCG@k
        if ndcg_metrics:
            print("\nNDCG@k:")
            for k in sorted(self.k_values):
                metric_name = f"ndcg@{k}"
                if metric_name in ndcg_metrics:
                    print(f"  {metric_name:15s}: {ndcg_metrics[metric_name]:.4f}")

        # Print other metrics
        if other_metrics:
            print("\nOther Metrics:")
            for metric_name, value in other_metrics.items():
                print(f"  {metric_name:15s}: {value:.4f}")

        print(f"{'=' * 60}\n")
