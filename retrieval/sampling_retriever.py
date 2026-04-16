"""
FAISS-based retrieval with random sampling strategy.
Retrieves top-n documents, randomly samples with probability p, then selects top-k from sampled set.
"""

from typing import List, Tuple, Optional
import numpy as np
import random

from retrieval.faiss_retriever import FAISSRetriever


class SamplingRetriever(FAISSRetriever):
    """
    FAISS retriever with random sampling from top-n results.

    Strategy:
    1. Retrieve top-n documents
    2. Randomly sample each document with probability p
    3. From sampled documents, select top-k by score
    """

    def __init__(self, encoder, seed: int = 42):
        """
        Args:
            encoder: Encoder model for embedding
            seed: Random seed for sampling reproducibility
        """
        super().__init__(encoder)
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def retrieve_with_sampling(
        self,
        queries: List[str],
        n: int = 100,
        p: float = 0.5,
        k: int = 10,
        batch_size: int = 32
    ) -> Tuple[List[List[str]], List[List[str]]]:
        """
        Retrieve with random sampling strategy.

        Args:
            queries: List of query strings
            n: Number of top documents to retrieve initially
            p: Sampling probability (0 < p < 1)
            k: Final number of documents to return after sampling
            batch_size: Batch size for query encoding

        Returns:
            Tuple of (baseline_results, sampled_results)
            - baseline_results: Standard top-k retrieval
            - sampled_results: Top-k after random sampling from top-n
        """
        if not (0 < p < 1):
            raise ValueError(f"Probability p must be between 0 and 1, got {p}")
        if k > n:
            raise ValueError(f"k ({k}) cannot be greater than n ({n})")

        # Get top-n results with scores
        results_with_scores = self.retrieve(queries, k=n, batch_size=batch_size)

        baseline_results = []
        sampled_results = []

        for query_results in results_with_scores:
            # Baseline: top-k directly
            baseline_titles = [
                self.passages[doc_id].title
                for doc_id, _ in query_results[:k]
            ]
            baseline_results.append(baseline_titles)

            # Sampling strategy
            # 1. Sample from top-n with probability p
            sampled_docs = []
            for doc_id, score in query_results:
                if random.random() < p:  # Sample with probability p
                    sampled_docs.append((doc_id, score))

            # 2. Sort sampled docs by score (descending)
            sampled_docs.sort(key=lambda x: x[1], reverse=True)

            # 3. Take top-k from sampled docs
            # If fewer than k docs were sampled, return all sampled docs
            top_k_sampled = sampled_docs[:k]

            sampled_titles = [
                self.passages[doc_id].title
                for doc_id, _ in top_k_sampled
            ]
            sampled_results.append(sampled_titles)

        return baseline_results, sampled_results

    def retrieve_batch_with_sampling(
        self,
        queries: List[str],
        n: int = 100,
        p: float = 0.5,
        k: int = 10,
        batch_size: int = 32
    ) -> Tuple[List[List[str]], List[List[str]], dict]:
        """
        Retrieve with sampling and return statistics.

        Args:
            queries: List of query strings
            n: Number of top documents to retrieve initially
            p: Sampling probability
            k: Final number of documents to return
            batch_size: Batch size for query encoding

        Returns:
            Tuple of (baseline_results, sampled_results, statistics)
        """
        baseline_results, sampled_results = self.retrieve_with_sampling(
            queries, n, p, k, batch_size
        )

        # Calculate statistics
        num_queries = len(queries)
        sampled_counts = [len(r) for r in sampled_results]

        statistics = {
            'n': n,
            'p': p,
            'k': k,
            'num_queries': num_queries,
            'avg_sampled_docs': np.mean(sampled_counts),
            'min_sampled_docs': np.min(sampled_counts),
            'max_sampled_docs': np.max(sampled_counts),
            'queries_with_less_than_k': sum(1 for c in sampled_counts if c < k),
            'expected_sample_size': n * p,
        }

        return baseline_results, sampled_results, statistics
