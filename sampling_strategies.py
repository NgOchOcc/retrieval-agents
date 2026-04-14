"""Sampling strategies for retrieval results."""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class SamplingConfig:
    """Configuration for sampling strategy."""

    strategy: str = "top_k"  # 'top_k', 'random_sample', 'diverse_sample'
    expansion_factor: int = 4  # Retrieve top (expansion_factor * k) documents
    random_ratio: float = 0.3  # Ratio of random sampling (0 < x < 1)
    seed: int = 42  # Random seed for reproducibility


class RetrievalSampler:
    """Apply sampling strategies to retrieval results."""

    def __init__(self, config: SamplingConfig = None):
        """
        Initialize retrieval sampler.

        Args:
            config: Sampling configuration
        """
        self.config = config or SamplingConfig()
        self.rng = np.random.RandomState(self.config.seed)

    def sample_top_k(
        self,
        doc_ids: List[str],
        scores: List[float],
        k: int,
    ) -> Tuple[List[str], List[float]]:
        """
        Standard top-k selection (no sampling).

        Args:
            doc_ids: List of document IDs (ranked)
            scores: List of scores (ranked)
            k: Number of documents to select

        Returns:
            Tuple of (selected_doc_ids, selected_scores)
        """
        return doc_ids[:k], scores[:k]

    def sample_random(
        self,
        doc_ids: List[str],
        scores: List[float],
        k: int,
    ) -> Tuple[List[str], List[float]]:
        """
        Random sampling with expansion and random selection.

        Process:
        1. Get top (expansion_factor * k) documents
        2. Randomly sample with ratio x from this pool
        3. Select top-k from sampled pool

        Args:
            doc_ids: List of document IDs (ranked)
            scores: List of scores (ranked)
            k: Number of documents to select

        Returns:
            Tuple of (selected_doc_ids, selected_scores)
        """
        # Step 1: Expand to top (expansion_factor * k)
        expanded_k = min(self.config.expansion_factor * k, len(doc_ids))
        expanded_doc_ids = doc_ids[:expanded_k]
        expanded_scores = scores[:expanded_k]

        # Step 2: Random sampling with ratio x
        num_to_sample = max(k, int(expanded_k * self.config.random_ratio))
        num_to_sample = min(num_to_sample, expanded_k)  # Don't exceed available docs

        # Random sample indices
        sampled_indices = self.rng.choice(
            expanded_k,
            size=num_to_sample,
            replace=False
        )

        # Get sampled docs and scores
        sampled_doc_ids = [expanded_doc_ids[i] for i in sampled_indices]
        sampled_scores = [expanded_scores[i] for i in sampled_indices]

        # Step 3: Sort by scores and select top-k
        # Combine and sort by score (descending)
        combined = list(zip(sampled_scores, sampled_doc_ids))
        combined.sort(reverse=True, key=lambda x: x[0])

        # Select top-k
        final_k = min(k, len(combined))
        final_scores = [score for score, _ in combined[:final_k]]
        final_doc_ids = [doc_id for _, doc_id in combined[:final_k]]

        return final_doc_ids, final_scores

    def sample_diverse(
        self,
        doc_ids: List[str],
        scores: List[float],
        k: int,
    ) -> Tuple[List[str], List[float]]:
        """
        Diverse sampling strategy.

        Process:
        1. Get top (expansion_factor * k) documents
        2. Use stratified sampling to ensure diversity
        3. Select top-k from sampled pool

        Args:
            doc_ids: List of document IDs (ranked)
            scores: List of scores (ranked)
            k: Number of documents to select

        Returns:
            Tuple of (selected_doc_ids, selected_scores)
        """
        # Step 1: Expand to top (expansion_factor * k)
        expanded_k = min(self.config.expansion_factor * k, len(doc_ids))
        expanded_doc_ids = doc_ids[:expanded_k]
        expanded_scores = scores[:expanded_k]

        # Step 2: Stratified sampling
        # Divide into bins and sample from each
        num_bins = min(4, expanded_k)  # Create 4 bins
        bin_size = expanded_k // num_bins

        sampled_indices = []
        for i in range(num_bins):
            start_idx = i * bin_size
            end_idx = start_idx + bin_size if i < num_bins - 1 else expanded_k

            # Sample from this bin
            bin_indices = list(range(start_idx, end_idx))
            num_from_bin = max(1, len(bin_indices) * self.config.random_ratio // num_bins)
            num_from_bin = int(min(num_from_bin, len(bin_indices)))

            sampled_from_bin = self.rng.choice(
                bin_indices,
                size=num_from_bin,
                replace=False
            )
            sampled_indices.extend(sampled_from_bin)

        # Get sampled docs and scores
        sampled_doc_ids = [expanded_doc_ids[i] for i in sampled_indices]
        sampled_scores = [expanded_scores[i] for i in sampled_indices]

        # Step 3: Sort and select top-k
        combined = list(zip(sampled_scores, sampled_doc_ids))
        combined.sort(reverse=True, key=lambda x: x[0])

        final_k = min(k, len(combined))
        final_scores = [score for score, _ in combined[:final_k]]
        final_doc_ids = [doc_id for _, doc_id in combined[:final_k]]

        return final_doc_ids, final_scores

    def apply_sampling(
        self,
        doc_ids: List[str],
        scores: List[float],
        k: int,
    ) -> Tuple[List[str], List[float]]:
        """
        Apply configured sampling strategy.

        Args:
            doc_ids: List of document IDs (ranked)
            scores: List of scores (ranked)
            k: Number of documents to select

        Returns:
            Tuple of (selected_doc_ids, selected_scores)
        """
        if self.config.strategy == "top_k":
            return self.sample_top_k(doc_ids, scores, k)
        elif self.config.strategy == "random_sample":
            return self.sample_random(doc_ids, scores, k)
        elif self.config.strategy == "diverse_sample":
            return self.sample_diverse(doc_ids, scores, k)
        else:
            raise ValueError(f"Unknown sampling strategy: {self.config.strategy}")

    def apply_to_batch(
        self,
        batch_doc_ids: List[List[str]],
        batch_scores: List[List[float]],
        k: int,
    ) -> Tuple[List[List[str]], List[List[float]]]:
        """
        Apply sampling to a batch of retrieval results.

        Args:
            batch_doc_ids: List of document ID lists
            batch_scores: List of score lists
            k: Number of documents to select per query

        Returns:
            Tuple of (sampled_doc_ids_batch, sampled_scores_batch)
        """
        sampled_doc_ids_batch = []
        sampled_scores_batch = []

        for doc_ids, scores in zip(batch_doc_ids, batch_scores):
            sampled_doc_ids, sampled_scores = self.apply_sampling(doc_ids, scores, k)
            sampled_doc_ids_batch.append(sampled_doc_ids)
            sampled_scores_batch.append(sampled_scores)

        return sampled_doc_ids_batch, sampled_scores_batch


def demonstrate_sampling():
    """Demonstrate different sampling strategies."""

    # Example retrieval results
    doc_ids = [f"doc_{i}" for i in range(100)]
    scores = [1.0 - i * 0.01 for i in range(100)]  # Decreasing scores

    k = 5

    print("=" * 60)
    print("Sampling Strategy Demonstration")
    print("=" * 60)
    print(f"\nInput: {len(doc_ids)} documents, k={k}")
    print(f"Top 20 scores: {scores[:20]}")

    # Strategy 1: Standard top-k
    print("\n" + "-" * 60)
    print("Strategy 1: Standard Top-K")
    print("-" * 60)
    config = SamplingConfig(strategy="top_k")
    sampler = RetrievalSampler(config)
    selected_ids, selected_scores = sampler.apply_sampling(doc_ids, scores, k)
    print(f"Selected docs: {selected_ids}")
    print(f"Selected scores: {selected_scores}")

    # Strategy 2: Random sampling
    print("\n" + "-" * 60)
    print("Strategy 2: Random Sampling (expansion_factor=4, random_ratio=0.3)")
    print("-" * 60)
    config = SamplingConfig(
        strategy="random_sample",
        expansion_factor=4,
        random_ratio=0.3,
        seed=42
    )
    sampler = RetrievalSampler(config)
    selected_ids, selected_scores = sampler.apply_sampling(doc_ids, scores, k)
    print(f"Selected docs: {selected_ids}")
    print(f"Selected scores: {selected_scores}")
    print(f"Process: top-{4*k} → random sample {int(4*k*0.3)} docs → top-{k}")

    # Strategy 3: Diverse sampling
    print("\n" + "-" * 60)
    print("Strategy 3: Diverse Sampling (stratified)")
    print("-" * 60)
    config = SamplingConfig(
        strategy="diverse_sample",
        expansion_factor=4,
        random_ratio=0.5,
        seed=42
    )
    sampler = RetrievalSampler(config)
    selected_ids, selected_scores = sampler.apply_sampling(doc_ids, scores, k)
    print(f"Selected docs: {selected_ids}")
    print(f"Selected scores: {selected_scores}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    demonstrate_sampling()
