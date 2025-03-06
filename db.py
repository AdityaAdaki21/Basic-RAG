# db.py

import faiss
import numpy as np

def build_index(embeddings):
    """
    Build a FAISS index from embeddings.
    """
    embeddings = np.array(embeddings).astype("float32")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def update_index(index, new_embeddings):
    """
    Update (insert) new embeddings into the existing FAISS index.
    """
    new_embeddings = np.array(new_embeddings).astype("float32")
    index.add(new_embeddings)
    return index

def search_index(query_embedding, index, top_k=3):
    """
    Search the FAISS index for the top_k most similar embeddings.
    """
    query_embedding = np.array(query_embedding).astype("float32")
    distances, indices = index.search(query_embedding, top_k)
    return distances, indices
