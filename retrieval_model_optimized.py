"""Optimized retrieval model wrapper for dense encoders with GPU acceleration and multi-processing."""

import torch
import numpy as np
from typing import List, Optional
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset
import torch.multiprocessing as mp

# Set multiprocessing start method to 'spawn' for CUDA compatibility
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set


class TextDataset(Dataset):
    """Dataset for text encoding with pre-tokenization."""

    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


class CollateFn:
    """Collate function for batching with tokenization (picklable for multiprocessing)."""

    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        """Tokenize batch on CPU (workers can do this safely)."""
        encoded = self.tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        # Return on CPU - will move to GPU in main process
        return encoded


class DenseRetrieverOptimized:
    """Optimized wrapper for dense retrieval models with GPU acceleration."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        normalize_embeddings: bool = True,
        max_length: int = 512,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        """
        Initialize optimized dense retriever.

        Args:
            model_name: HuggingFace model name or path
            device: Device to use ('cuda' or 'cpu')
            normalize_embeddings: Whether to normalize embeddings
            max_length: Maximum sequence length
            batch_size: Batch size for encoding
            num_workers: Number of workers for data loading (0 = main process only)
            pin_memory: Use pinned memory for faster GPU transfer
        """
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.normalize_embeddings = normalize_embeddings
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers if self.device == "cuda" else 0
        self.pin_memory = pin_memory and self.device == "cuda"

        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Enable cuDNN autotuner for better performance
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True

        # Check if model needs special instructions (e.g., E5 models)
        self.needs_instruction = "e5" in model_name.lower()

        print(f"Model loaded on {self.device}")
        if self.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Batch size: {batch_size}, Num workers: {num_workers}")
            print(f"Pin memory: {pin_memory}")

    def encode(
        self,
        texts: List[str],
        is_query: bool = False,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode texts to embeddings with optimized batching.

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

        # Create dataset and dataloader
        dataset = TextDataset(texts, self.tokenizer, self.max_length)

        # Use DataLoader for efficient batching and multi-processing
        # Workers do tokenization on CPU, main process moves to GPU
        # Use class-based collate_fn (picklable for spawn method)
        collate_fn = CollateFn(self.tokenizer, self.max_length)

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,  # Keep workers alive
            collate_fn=collate_fn,
        )

        all_embeddings = []

        # Wrap dataloader with tqdm
        if show_progress:
            dataloader = tqdm(
                dataloader,
                desc=f"Encoding {'queries' if is_query else 'documents'}",
                total=len(dataloader),
            )

        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device (from CPU/pinned memory to GPU)
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Get embeddings
                outputs = self.model(**batch)

                # Use mean pooling
                embeddings = self._pool_embeddings(outputs, batch["attention_mask"])

                # Normalize if required
                if self.normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                # Move to CPU and append
                all_embeddings.append(embeddings.cpu().numpy())

        # Concatenate all batches
        all_embeddings = np.vstack(all_embeddings)
        return all_embeddings

    def _pool_embeddings(self, outputs, attention_mask):
        """
        Pool token embeddings to get sentence embeddings using mean pooling.

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

    def auto_batch_size(self, num_samples: int = 100) -> int:
        """
        Automatically determine optimal batch size based on GPU memory.

        Args:
            num_samples: Number of samples to test

        Returns:
            Optimal batch size
        """
        if self.device == "cpu":
            return self.batch_size

        # Test different batch sizes
        test_text = "This is a test sentence for batch size optimization."
        test_texts = [test_text] * num_samples

        batch_sizes = [16, 32, 64, 128, 256, 512]
        optimal_batch_size = self.batch_size

        print("Auto-detecting optimal batch size...")

        for bs in batch_sizes:
            try:
                # Clear cache
                if self.device == "cuda":
                    torch.cuda.empty_cache()

                # Test encoding with this batch size
                old_batch_size = self.batch_size
                self.batch_size = bs

                _ = self.encode(test_texts[:bs], show_progress=False)

                optimal_batch_size = bs
                print(f"  Batch size {bs}: ✓")

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  Batch size {bs}: ✗ (OOM)")
                    break
                else:
                    raise e
            finally:
                self.batch_size = old_batch_size

        # Use 80% of max to be safe
        if optimal_batch_size > 16:
            optimal_batch_size = int(optimal_batch_size * 0.8)

        print(f"Optimal batch size: {optimal_batch_size}")
        return optimal_batch_size


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
