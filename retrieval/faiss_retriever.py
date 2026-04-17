"""
FAISS-based retrieval engine for dense retrieval.
Uses IndexFlatIP for exact cosine similarity search with GPU support and caching.
"""

from typing import List, Tuple, Optional
import numpy as np
import faiss
import torch
import pickle
from pathlib import Path
from tqdm import tqdm

from data import Passage
from models import BaseEncoder


class FAISSRetriever:
    """FAISS-based dense retriever with GPU support and caching."""

    def __init__(self, encoder: BaseEncoder, use_gpu: bool = True):
        """
        Args:
            encoder: Encoder model for embedding
            use_gpu: Use GPU for FAISS index (default: True)
        """
        self.encoder = encoder
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.index: Optional[faiss.Index] = None
        self.passages: List[Passage] = []
        self.passage_embeddings: Optional[np.ndarray] = None

        if self.use_gpu:
            self.gpu_resource = faiss.StandardGpuResources()
            print(f"FAISS GPU initialized")

    def build_index(
        self,
        passages: List[Passage],
        batch_size: int = 128,
        save_embeddings: bool = True,
        cache_dir: Optional[str] = None
    ):
        """
        Build FAISS index from passages with caching support.

        Args:
            passages: List of passages to index
            batch_size: Batch size for encoding
            save_embeddings: Whether to store embeddings in memory
            cache_dir: Directory to cache index (None = no caching)
        """
        # Try load from cache
        if cache_dir:
            loaded = self._load_from_cache(cache_dir, passages)
            if loaded:
                print(f"✓ Index loaded from cache ({self.index.ntotal} vectors)")
                return

        print(f"Building index for {len(passages)} passages...")
        self.passages = passages

        # Format passages
        formatted_passages = [
            self.encoder.format_passage(p.text) for p in passages
        ]

        # Encode passages
        embeddings = self.encoder.encode(
            formatted_passages,
            batch_size=batch_size,
            normalize=True,
            show_progress=True
        )

        # Store embeddings if requested
        if save_embeddings:
            self.passage_embeddings = embeddings

        # Build FAISS index
        embedding_dim = embeddings.shape[1]
        cpu_index = faiss.IndexFlatIP(embedding_dim)
        cpu_index.add(embeddings.astype(np.float32))

        # Move to GPU if enabled
        if self.use_gpu:
            self.index = faiss.index_cpu_to_gpu(self.gpu_resource, 0, cpu_index)
            print(f"✓ Index built with GPU ({self.index.ntotal} vectors)")
        else:
            self.index = cpu_index
            print(f"✓ Index built with CPU ({self.index.ntotal} vectors)")

        # Save to cache
        if cache_dir:
            self._save_to_cache(cache_dir, cpu_index, passages)

    def retrieve(
        self,
        queries: List[str],
        k: int = 10,
        batch_size: int = 32
    ) -> List[List[Tuple[int, float]]]:
        """
        Retrieve top-k passages for each query.

        Args:
            queries: List of query strings
            k: Number of passages to retrieve per query
            batch_size: Batch size for query encoding

        Returns:
            List of lists, where each inner list contains (doc_id, score) tuples
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        # Format queries with instruction
        formatted_queries = [
            self.encoder.format_query(q) for q in queries
        ]

        # Encode queries (normalized)
        query_embeddings = self.encoder.encode(
            formatted_queries,
            batch_size=batch_size,
            normalize=True,
            show_progress=True
        )

        # Search in FAISS index
        scores, indices = self.index.search(
            query_embeddings.astype(np.float32), k
        )

        # Convert to list of (doc_id, score) tuples
        results = []
        for i in range(len(queries)):
            query_results = [
                (int(indices[i, j]), float(scores[i, j]))
                for j in range(k)
                if indices[i, j] != -1  # Filter out padding
            ]
            results.append(query_results)

        return results

    def retrieve_batch(
        self,
        queries: List[str],
        k: int = 10,
        batch_size: int = 32
    ) -> List[List[str]]:
        """
        Retrieve top-k passage titles for each query.

        Args:
            queries: List of query strings
            k: Number of passages to retrieve per query
            batch_size: Batch size for query encoding

        Returns:
            List of lists of retrieved passage titles
        """
        # Get doc IDs and scores
        results = self.retrieve(queries, k, batch_size)

        # Map to titles
        retrieved_titles = []
        for query_results in results:
            titles = [
                self.passages[doc_id].title
                for doc_id, _ in query_results
            ]
            retrieved_titles.append(titles)

        return retrieved_titles

    def get_passage_by_id(self, doc_id: int) -> Passage:
        """Get passage by document ID."""
        return self.passages[doc_id]

    def _get_cache_path(self, cache_dir: str) -> Path:
        """Get cache file path."""
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        model_name = self.encoder.model_name.replace("/", "_")
        return cache_path / f"{model_name}_index.faiss"

    def _get_passages_cache_path(self, cache_dir: str) -> Path:
        """Get passages cache file path."""
        cache_path = Path(cache_dir)
        model_name = self.encoder.model_name.replace("/", "_")
        return cache_path / f"{model_name}_passages.pkl"

    def _save_to_cache(self, cache_dir: str, cpu_index: faiss.Index, passages: List[Passage]):
        """Save index and passages to cache."""
        try:
            index_path = self._get_cache_path(cache_dir)
            passages_path = self._get_passages_cache_path(cache_dir)

            faiss.write_index(cpu_index, str(index_path))
            with open(passages_path, 'wb') as f:
                pickle.dump(passages, f)

            print(f"✓ Cached to {index_path}")
        except Exception as e:
            print(f"⚠ Cache save failed: {e}")

    def _load_from_cache(self, cache_dir: str, passages: List[Passage]) -> bool:
        """Load index and passages from cache."""
        try:
            index_path = self._get_cache_path(cache_dir)
            passages_path = self._get_passages_cache_path(cache_dir)

            if not index_path.exists() or not passages_path.exists():
                return False

            # Load passages
            with open(passages_path, 'rb') as f:
                cached_passages = pickle.load(f)

            # Verify passages match
            if len(cached_passages) != len(passages):
                print("⚠ Cache mismatch (size), rebuilding...")
                return False

            if (cached_passages[0].text != passages[0].text or
                cached_passages[-1].text != passages[-1].text):
                print("⚠ Cache mismatch (content), rebuilding...")
                return False

            # Load FAISS index
            cpu_index = faiss.read_index(str(index_path))

            # Move to GPU if enabled
            if self.use_gpu:
                self.index = faiss.index_cpu_to_gpu(self.gpu_resource, 0, cpu_index)
            else:
                self.index = cpu_index

            self.passages = cached_passages
            return True

        except Exception as e:
            print(f"⚠ Cache load failed: {e}")
            return False
