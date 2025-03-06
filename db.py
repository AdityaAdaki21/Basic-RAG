# db.py

import faiss
import numpy as np
import os
import pickle
from typing import Tuple, List, Optional, Dict, Any

class VectorStore:
    """Manages vector storage and retrieval using FAISS."""
    
    def __init__(self, dimension: Optional[int] = None, index_type: str = "flat"):
        """
        Initialize the vector store.
        
        Args:
            dimension: Dimension of the vectors (required if creating a new index)
            index_type: Type of FAISS index to use ('flat', 'ivf', or 'hnsw')
        """
        self.index = None
        self.dimension = dimension
        self.index_type = index_type
        self.metadata = {
            "total_vectors": 0,
            "index_type": index_type,
            "chunks": []  # Store text chunks corresponding to embeddings
        }
    
    def create_index(self, dimension: Optional[int] = None) -> None:
        """
        Create a new FAISS index.
        
        Args:
            dimension: Dimension of the vectors (required if not provided in constructor)
        """
        dim = dimension or self.dimension
        if dim is None:
            raise ValueError("Vector dimension must be specified")
        
        self.dimension = dim
        
        if self.index_type == "flat":
            # Simple but effective for smaller datasets
            self.index = faiss.IndexFlatL2(dim)
        elif self.index_type == "ivf":
            # Better for larger datasets
            quantizer = faiss.IndexFlatL2(dim)
            nlist = 100  # number of clusters
            self.index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
            self.index.train(np.random.random((1000, dim)).astype(np.float32))  # Need to train with some data
        elif self.index_type == "hnsw":
            # Best for approximate nearest neighbor search
            self.index = faiss.IndexHNSWFlat(dim, 32)  # 32 is the number of neighbors
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
            
        self.metadata["total_vectors"] = 0
    
    def add_embeddings(self, embeddings: np.ndarray, chunks: Optional[List[str]] = None) -> None:
        """
        Add embeddings to the index.
        
        Args:
            embeddings: Array of embeddings to add
            chunks: Corresponding text chunks for these embeddings
        """
        if self.index is None:
            self.create_index(embeddings.shape[1])
            
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings).astype(np.float32)
            
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
            
        self.index.add(embeddings)
        self.metadata["total_vectors"] += embeddings.shape[0]
        
        # Store corresponding text chunks if provided
        if chunks:
            self.metadata["chunks"].extend(chunks)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search the index for similar vectors.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            
        Returns:
            Tuple of (distances, indices)
        """
        if self.index is None:
            raise ValueError("Index has not been created or loaded")
            
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding).astype(np.float32)
            
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)
            
        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        # Limit top_k to the number of vectors in the index
        effective_top_k = min(top_k, self.metadata["total_vectors"])
        
        distances, indices = self.index.search(query_embedding, effective_top_k)
        return distances, indices
    
    def get_chunks_by_indices(self, indices: List[int]) -> List[str]:
        """
        Retrieve the text chunks corresponding to the given indices.
        
        Args:
            indices: List of indices
            
        Returns:
            List of corresponding text chunks
        """
        if not self.metadata.get("chunks"):
            raise ValueError("No text chunks are stored in the index")
            
        return [self.metadata["chunks"][i] for i in indices if i < len(self.metadata["chunks"])]
    
    def save(self, directory: str, filename: str = "vector_store") -> str:
        """
        Save the index and metadata to disk.
        
        Args:
            directory: Directory to save to
            filename: Base filename to use
            
        Returns:
            Path to the saved index
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save the index
        index_path = os.path.join(directory, f"{filename}.index")
        faiss.write_index(self.index, index_path)
        
        # Save the metadata
        metadata_path = os.path.join(directory, f"{filename}.meta")
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
            
        return index_path
    
    def load(self, directory: str, filename: str = "vector_store") -> None:
        """
        Load the index and metadata from disk.
        
        Args:
            directory: Directory to load from
            filename: Base filename to use
        """
        # Load the index
        index_path = os.path.join(directory, f"{filename}.index")
        self.index = faiss.read_index(index_path)
        self.dimension = self.index.d
        
        # Load the metadata
        metadata_path = os.path.join(directory, f"{filename}.meta")
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)

# For backward compatibility
def build_index(embeddings):
    """Legacy function for backward compatibility."""
    store = VectorStore()
    store.create_index(embeddings.shape[1])
    store.add_embeddings(embeddings)
    return store.index

def update_index(index, new_embeddings):
    """Legacy function for backward compatibility."""
    new_embeddings = np.array(new_embeddings).astype("float32")
    index.add(new_embeddings)
    return index

def search_index(query_embedding, index, top_k=6):
    """Legacy function for backward compatibility."""
    query_embedding = np.array(query_embedding).astype("float32")
    distances, indices = index.search(query_embedding, top_k)
    return distances, indices