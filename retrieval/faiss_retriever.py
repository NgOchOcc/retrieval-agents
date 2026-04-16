"""
FAISS-based retrieval engine for dense retrieval.
Uses IndexFlatIP for exact cosine similarity search.
"""

from typing import List, Tuple, Optional
import numpy as np
import faiss
from tqdm import tqdm

from data import Passage
from models import BaseEncoder


class FAISSRetriever:
    """FAISS-based dense retriever."""

    def __init__(self, encoder: BaseEncoder):
        """
        Args:
            encoder: Encoder model for embedding
        """
        self.encoder = encoder
        self.index: Optional[faiss.Index] = None
        self.passages: List[Passage] = []
        self.passage_embeddings: Optional[np.ndarray] = None

    def build_index(
        self,
        passages: List[Passage],
        batch_size: int = 128,
        save_embeddings: bool = True
    ):
        """
        Build FAISS index from passages.

        Args:
            passages: List of passages to index
            batch_size: Batch size for encoding
            save_embeddings: Whether to store embeddings in memory
        """
        print(f"Building index for {len(passages)} passages...")
        self.passages = passages

        # Format passages with instruction
        formatted_passages = [
            self.encoder.format_passage(p.text) for p in passages
        ]

        # Encode passages (normalized)
        embeddings = self.encoder.encode(
            formatted_passages,
            batch_size=batch_size,
            normalize=True,
            show_progress=True
        )

        # Store embeddings if requested
        if save_embeddings:
            self.passage_embeddings = embeddings

        # Build FAISS index (IndexFlatIP for cosine similarity on normalized vectors)
        embedding_dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(embedding_dim)

        # Add embeddings to index
        self.index.add(embeddings.astype(np.float32))

        print(f"Index built with {self.index.ntotal} vectors")

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
