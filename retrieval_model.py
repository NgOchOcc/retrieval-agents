"""Retrieval model wrapper for dense encoders."""

import torch
import numpy as np
from typing import List, Union, Optional
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


class DenseRetriever:
    """Wrapper for dense retrieval models (BGE, E5, GTE, etc.)."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        normalize_embeddings: bool = True,
        max_length: int = 512,
        batch_size: int = 32,
    ):
        """
        Initialize dense retriever.

        Args:
            model_name: HuggingFace model name or path
            device: Device to use ('cuda' or 'cpu')
            normalize_embeddings: Whether to normalize embeddings
            max_length: Maximum sequence length
            batch_size: Batch size for encoding
        """
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.normalize_embeddings = normalize_embeddings
        self.max_length = max_length
        self.batch_size = batch_size

        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Check if model needs special instructions (e.g., E5 models)
        self.needs_instruction = "e5" in model_name.lower()

        print(f"Model loaded on {self.device}")

    def encode(
        self,
        texts: List[str],
        is_query: bool = False,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode texts to embeddings.

        Args:
            texts: List of texts to encode
            is_query: Whether texts are queries (for models that need query prefix)
            show_progress: Whether to show progress bar

        Returns:
            Numpy array of embeddings [num_texts, embedding_dim]
        """
        # Add instruction prefix for E5 models
        if self.needs_instruction:
            if is_query:
                texts = [f"query: {text}" for text in texts]
            else:
                texts = [f"passage: {text}" for text in texts]

        all_embeddings = []

        # Process in batches
        num_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        iterator = range(0, len(texts), self.batch_size)

        if show_progress:
            iterator = tqdm(
                iterator, desc=f"Encoding {'queries' if is_query else 'documents'}", total=num_batches
            )

        with torch.no_grad():
            for i in iterator:
                batch_texts = texts[i : i + self.batch_size]

                # Tokenize
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Get embeddings
                outputs = self.model(**inputs)

                # Use mean pooling or CLS token
                embeddings = self._pool_embeddings(outputs, inputs["attention_mask"])

                # Normalize if required
                if self.normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                all_embeddings.append(embeddings.cpu().numpy())

        # Concatenate all batches
        all_embeddings = np.vstack(all_embeddings)
        return all_embeddings

    def _pool_embeddings(self, outputs, attention_mask):
        """
        Pool token embeddings to get sentence embeddings.

        Args:
            outputs: Model outputs
            attention_mask: Attention mask

        Returns:
            Pooled embeddings
        """
        # Mean pooling
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode_queries(self, queries: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Encode queries.

        Args:
            queries: List of query strings
            show_progress: Whether to show progress bar

        Returns:
            Query embeddings
        """
        return self.encode(queries, is_query=True, show_progress=show_progress)

    def encode_corpus(self, corpus: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Encode corpus documents.

        Args:
            corpus: List of document strings
            show_progress: Whether to show progress bar

        Returns:
            Document embeddings
        """
        return self.encode(corpus, is_query=False, show_progress=show_progress)

    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        # Encode a dummy text to get dimension
        dummy_embedding = self.encode(["test"], show_progress=False)
        return dummy_embedding.shape[1]


class RetrievalResult:
    """Container for retrieval results."""

    def __init__(self, doc_ids: List[str], scores: List[float], doc_texts: Optional[List[str]] = None):
        """
        Initialize retrieval result.

        Args:
            doc_ids: List of retrieved document IDs
            scores: List of retrieval scores
            doc_texts: Optional list of document texts
        """
        self.doc_ids = doc_ids
        self.scores = scores
        self.doc_texts = doc_texts

    def top_k(self, k: int) -> "RetrievalResult":
        """Get top-k results."""
        return RetrievalResult(
            doc_ids=self.doc_ids[:k],
            scores=self.scores[:k],
            doc_texts=self.doc_texts[:k] if self.doc_texts else None,
        )
