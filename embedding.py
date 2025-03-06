# embedding.py

import requests
import json
import numpy as np
import time
from functools import lru_cache

class EmbeddingClient:
    """Client for generating embeddings from Ollama API."""
    
    def __init__(self, base_url="http://localhost:11434", default_model="mxbai-embed-large"):
        self.base_url = base_url.rstrip('/')
        self.default_model = default_model
        self.embedding_endpoint = f"{self.base_url}/api/embeddings"
        self.model_endpoint = f"{self.base_url}/api/show"
        self.loaded_model = None
        self.ensure_model_loaded(default_model)  # Load the model when client is initialized

    def ensure_model_loaded(self, model):
        """Ensure the specified model is loaded and ready for use."""
        if model == self.loaded_model:
            return  # Model is already loaded
        
        try:
            # Check if the model is loaded
            response = requests.post(
                self.model_endpoint,
                json={"name": model},
                timeout=(5, 10)
            )
            
            if response.status_code == 200:
                self.loaded_model = model
                print(f"Model {model} is already loaded")
            else:
                # If not loaded, load it
                print(f"Loading model {model}...")
                response = requests.post(
                    f"{self.base_url}/api/pull",
                    json={"name": model},
                    timeout=(5, 300)  # Longer timeout for model loading
                )
                response.raise_for_status()
                self.loaded_model = model
                print(f"Model {model} successfully loaded")
        except Exception as e:
            print(f"Failed to ensure model is loaded: {e}")
            # Continue anyway, the embeddings endpoint will load the model if needed"
    
    def embed(self, text, model=None, max_retries=3, retry_delay=1):
        """
        Generate embeddings for the provided text with retry logic.
        
        Args:
            text: Text to embed
            model: Model to use (defaults to instance default_model)
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            
        Returns:
            List of embedding values
        """
        if not text.strip():
            raise ValueError("Empty text provided for embedding")
            
        model = model or self.default_model
        
        # Ensure the model is loaded
        if model != self.loaded_model:
            self.ensure_model_loaded(model)

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.embedding_endpoint,
                    json={"model": model, "prompt": text},
                    timeout=(5, 60)  # (connect_timeout, read_timeout)
                )
                
                response.raise_for_status()  # Raise exception for HTTP errors
                
                result = response.json()
                if "embedding" in result:
                    return result["embedding"]
                else:
                    raise ValueError(f"Embedding not found in response: {result}")
                    
            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    print(f"API request failed (attempt {attempt+1}/{max_retries}): {e}")
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    raise Exception(f"Failed to generate embedding after {max_retries} attempts: {e}")
    
    def compute_batch_embeddings(self, chunks, batch_size=5, model=None, progress_callback=None):
        """
        Compute embeddings for a list of text chunks with batching.
        
        Args:
            chunks: List of text chunks
            batch_size: Number of chunks to process in parallel
            model: Model to use
            progress_callback: Function to call with progress updates
            
        Returns:
            List of embeddings
        """
        model = model or self.default_model
        embeddings = []
        
        # Process chunks in batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            batch_embeddings = []
            
            # Process each chunk in the batch
            for j, chunk in enumerate(batch):
                chunk_idx = i + j
                if progress_callback:
                    progress_callback(chunk_idx, len(chunks))
                else:
                    print(f"Processing chunk {chunk_idx+1}/{len(chunks)}")
                
                try:
                    embedding = self.embed(chunk, model)
                    batch_embeddings.append(embedding)
                except Exception as e:
                    print(f"Error processing chunk {chunk_idx+1}: {e}")
                    # Instead of skipping, use a zero vector of the correct size
                    # First, check if we have other embeddings to get the size from
                    if embeddings or batch_embeddings:
                        dim = len(embeddings[0] if embeddings else batch_embeddings[0])
                        batch_embeddings.append([0.0] * dim)
                    else:
                        # If this is the first chunk and it failed, we have a problem
                        raise Exception("Failed to generate any embeddings to determine vector size")
            
            embeddings.extend(batch_embeddings)
        
        return embeddings

# For backward compatibility
def ollama_embed(text, model="mxbai-embed-large"):
    """Legacy function for backward compatibility."""
    client = EmbeddingClient()
    return client.embed(text, model)

def compute_embeddings_ollama(chunks, model="mxbai-embed-large"):
    """Legacy function for backward compatibility."""
    client = EmbeddingClient()
    return client.compute_batch_embeddings(chunks, model=model)