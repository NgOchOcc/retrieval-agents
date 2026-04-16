"""FAISS-based indexing for efficient retrieval."""

import os
import numpy as np
import faiss
import pickle
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

from retrieval_model import RetrievalResult


class FAISSIndexer:
    """FAISS indexer for dense retrieval."""

    def __init__(
        self,
        embedding_dim: int,
        index_type: str = "Flat",
        use_gpu: bool = False,
        normalize: bool = True,
        gpu_id: int = 0,
    ):
        """
        Initialize FAISS indexer.

        Args:
            embedding_dim: Dimension of embeddings
            index_type: Type of FAISS index ('Flat' or 'IVF')
            use_gpu: Whether to use GPU for index
            normalize: Whether to normalize embeddings (for cosine similarity)
            gpu_id: GPU device ID to use (default: 0)
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.use_gpu = use_gpu
        self.normalize = normalize
        self.gpu_id = gpu_id

        self.index = None
        self.doc_ids = []
        self.gpu_resources = None
        self.cpu_index = None  # Keep CPU index reference for IVF training

        # Check GPU availability
        if self.use_gpu:
            num_gpus = faiss.get_num_gpus()
            if num_gpus == 0:
                print("⚠️  WARNING: GPU requested but no GPU available. Using CPU instead.")
                self.use_gpu = False
            else:
                print(f"✅ GPU indexing enabled (GPUs available: {num_gpus}, using GPU {gpu_id})")
                # Create persistent GPU resources with more memory
                self.gpu_resources = faiss.StandardGpuResources()
                # Increase temp memory for large datasets (H100 has plenty)
                self.gpu_resources.setTempMemory(2 * 1024 * 1024 * 1024)  # 2GB temp memory
                print(f"✅ GPU resources allocated (2GB temp memory)")

        print(f"Initializing FAISS index (type={index_type}, dim={embedding_dim}, gpu={self.use_gpu})")
        self._create_index()

    def _create_index(self):
        """Create FAISS index."""
        # Create CPU index first
        cpu_index = None

        if self.index_type == "Flat":
            # Flat index (exact search)
            if self.normalize:
                # Use inner product for normalized vectors (equivalent to cosine similarity)
                cpu_index = faiss.IndexFlatIP(self.embedding_dim)
            else:
                cpu_index = faiss.IndexFlatL2(self.embedding_dim)

        elif self.index_type == "IVF":
            # IVF index (approximate search, faster for large corpora)
            nlist = 4096  # number of clusters (increased for better accuracy)
            quantizer = faiss.IndexFlatL2(self.embedding_dim)

            if self.normalize:
                cpu_index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT)
            else:
                cpu_index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist, faiss.METRIC_L2)

        elif self.index_type == "IVFPQ":
            # IVF with Product Quantization (more memory efficient)
            nlist = 4096
            m = 64  # number of subquantizers
            quantizer = faiss.IndexFlatL2(self.embedding_dim)

            if self.normalize:
                cpu_index = faiss.IndexIVFPQ(quantizer, self.embedding_dim, nlist, m, 8, faiss.METRIC_INNER_PRODUCT)
            else:
                cpu_index = faiss.IndexIVFPQ(quantizer, self.embedding_dim, nlist, m, 8)

        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")

        # Store CPU index for IVF training
        self.cpu_index = cpu_index

        # Move to GPU if requested
        if self.use_gpu and self.gpu_resources is not None:
            print(f"🚀 Moving index to GPU {self.gpu_id}...")
            try:
                # Use the persistent GPU resources
                self.index = faiss.index_cpu_to_gpu(self.gpu_resources, self.gpu_id, cpu_index)
                print(f"✅ Index successfully moved to GPU {self.gpu_id}")
                print(f"✅ Index type on GPU: {type(self.index).__name__}")
            except Exception as e:
                print(f"❌ Failed to move index to GPU: {e}")
                print("⚠️  Falling back to CPU index")
                self.index = cpu_index
                self.use_gpu = False
        else:
            self.index = cpu_index
            if not self.use_gpu:
                print("ℹ️  Using CPU index")

    def add_documents(self, embeddings: np.ndarray, doc_ids: List[str]):
        """
        Add documents to index.

        Args:
            embeddings: Document embeddings [num_docs, embedding_dim]
            doc_ids: Document IDs
        """
        assert len(embeddings) == len(doc_ids), "Mismatch between embeddings and doc_ids"

        print(f"Adding {len(embeddings):,} documents to index")

        # Normalize if required
        if self.normalize:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Convert to float32 (FAISS requirement)
        embeddings = embeddings.astype(np.float32)

        # Train index if needed (for IVF)
        if self.index_type in ["IVF", "IVFPQ"]:
            if not self.index.is_trained:
                print(f"🎓 Training {self.index_type} index on {'GPU' if self.use_gpu else 'CPU'}...")

                # For GPU index, train on CPU first then move to GPU
                if self.use_gpu:
                    # Train on CPU index
                    self.cpu_index.train(embeddings)
                    print(f"✅ Training completed on CPU")

                    # Rebuild GPU index with trained CPU index
                    print(f"🚀 Moving trained index to GPU {self.gpu_id}...")
                    self.index = faiss.index_cpu_to_gpu(self.gpu_resources, self.gpu_id, self.cpu_index)
                    print(f"✅ Trained index moved to GPU")
                else:
                    # Train on CPU
                    self.index.train(embeddings)
                    print(f"✅ Training completed")

        # Add to index
        print(f"📥 Adding embeddings to {'GPU' if self.use_gpu else 'CPU'} index...")
        self.index.add(embeddings)
        self.doc_ids.extend(doc_ids)

        print(f"✅ Index now contains {self.index.ntotal:,} documents on {'GPU' if self.use_gpu else 'CPU'}")

    def search(
        self,
        query_embeddings: np.ndarray,
        k: int = 10,
        return_scores: bool = True,
    ) -> List[RetrievalResult]:
        """
        Search index for nearest neighbors.

        Args:
            query_embeddings: Query embeddings [num_queries, embedding_dim]
            k: Number of top results to return
            return_scores: Whether to return similarity scores

        Returns:
            List of RetrievalResult for each query
        """
        # Normalize if required
        if self.normalize:
            query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)

        # Convert to float32
        query_embeddings = query_embeddings.astype(np.float32)

        # Search (automatically uses GPU if index is on GPU)
        scores, indices = self.index.search(query_embeddings, k)

        # Convert to results
        results = []
        for i in range(len(query_embeddings)):
            doc_ids = [self.doc_ids[idx] for idx in indices[i] if idx < len(self.doc_ids)]
            score_list = scores[i][: len(doc_ids)].tolist() if return_scores else None

            result = RetrievalResult(doc_ids=doc_ids, scores=score_list)
            results.append(result)

        return results

    def save(self, save_dir: str):
        """
        Save index to disk.

        Args:
            save_dir: Directory to save index
        """
        os.makedirs(save_dir, exist_ok=True)

        # Save FAISS index
        index_path = os.path.join(save_dir, "index.faiss")
        if self.use_gpu:
            # Move to CPU before saving
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, index_path)
        else:
            faiss.write_index(self.index, index_path)

        # Save doc IDs
        doc_ids_path = os.path.join(save_dir, "doc_ids.pkl")
        with open(doc_ids_path, "wb") as f:
            pickle.dump(self.doc_ids, f)

        # Save config
        config_path = os.path.join(save_dir, "config.pkl")
        config = {
            "embedding_dim": self.embedding_dim,
            "index_type": self.index_type,
            "normalize": self.normalize,
        }
        with open(config_path, "wb") as f:
            pickle.dump(config, f)

        print(f"Index saved to {save_dir}")

    def load(self, save_dir: str):
        """
        Load index from disk.

        Args:
            save_dir: Directory containing saved index
        """
        # Load config
        config_path = os.path.join(save_dir, "config.pkl")
        with open(config_path, "rb") as f:
            config = pickle.load(f)

        self.embedding_dim = config["embedding_dim"]
        self.index_type = config["index_type"]
        self.normalize = config["normalize"]

        # Load FAISS index (always loads to CPU first)
        index_path = os.path.join(save_dir, "index.faiss")
        cpu_index = faiss.read_index(index_path)
        self.cpu_index = cpu_index

        # Move to GPU if requested
        if self.use_gpu and self.gpu_resources is not None:
            print(f"🚀 Moving loaded index to GPU {self.gpu_id}...")
            try:
                self.index = faiss.index_cpu_to_gpu(self.gpu_resources, self.gpu_id, cpu_index)
                print(f"✅ Index successfully moved to GPU {self.gpu_id}")
                print(f"✅ Index type on GPU: {type(self.index).__name__}")
            except Exception as e:
                print(f"❌ Failed to move index to GPU: {e}")
                print("⚠️  Using CPU index instead")
                self.index = cpu_index
                self.use_gpu = False
        else:
            self.index = cpu_index

        # Load doc IDs
        doc_ids_path = os.path.join(save_dir, "doc_ids.pkl")
        with open(doc_ids_path, "rb") as f:
            self.doc_ids = pickle.load(f)

        print(f"✅ Index loaded from {save_dir} with {self.index.ntotal:,} documents on {'GPU' if self.use_gpu else 'CPU'}")

    def get_num_documents(self) -> int:
        """Get number of documents in index."""
        return self.index.ntotal if self.index else 0


class SimpleRetriever:
    """Simple numpy-based retriever (no FAISS, for small corpora)."""

    def __init__(self, normalize: bool = True):
        """
        Initialize simple retriever.

        Args:
            normalize: Whether to normalize embeddings
        """
        self.normalize = normalize
        self.embeddings = None
        self.doc_ids = []

    def add_documents(self, embeddings: np.ndarray, doc_ids: List[str]):
        """Add documents."""
        assert len(embeddings) == len(doc_ids)

        if self.normalize:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        self.embeddings = embeddings.astype(np.float32)
        self.doc_ids = doc_ids

    def search(self, query_embeddings: np.ndarray, k: int = 10) -> List[RetrievalResult]:
        """Search for nearest neighbors using cosine similarity."""
        if self.normalize:
            query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)

        # Compute cosine similarity
        scores = np.dot(query_embeddings, self.embeddings.T)

        # Get top-k
        results = []
        for i in range(len(query_embeddings)):
            top_indices = np.argsort(scores[i])[::-1][:k]
            top_scores = scores[i][top_indices]

            doc_ids = [self.doc_ids[idx] for idx in top_indices]
            result = RetrievalResult(doc_ids=doc_ids, scores=top_scores.tolist())
            results.append(result)

        return results
