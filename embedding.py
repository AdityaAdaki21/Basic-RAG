# embedding.py

import requests
import json

def ollama_embed(text, model="mxbai-embed-large"):
    """
    Calls the Ollama API to generate embeddings for the provided text.
    Uses the embeddings endpoint instead of run command.
    """
    try:
        # Use the Ollama API instead of CLI
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": model, "prompt": text}
        )
        
        if response.status_code != 200:
            print(f"Ollama API error: {response.text}")
            raise Exception(f"Ollama API returned status code {response.status_code}")
            
        result = response.json()
        if "embedding" in result:
            return result["embedding"]
        else:
            print(f"Unexpected response format: {result}")
            raise Exception("Embedding not found in response")
            
    except Exception as e:
        print(f"Error calling Ollama embeddings API: {e}")
        raise Exception("Error calling Ollama embed model: " + str(e))

def compute_embeddings_ollama(chunks, model="mxbai-embed-large"):
    """
    Compute embeddings for a list of text chunks using the Ollama embedding model.
    """
    embeddings = []
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}")
        try:
            embedding = ollama_embed(chunk, model)
            embeddings.append(embedding)
        except Exception as e:
            print(f"Error processing chunk {i+1}: {e}")
            print("Attempting to continue with other chunks...")
            # Return an empty embedding or skip this chunk
            # You might want to use a strategy like retrying or breaking it down into smaller chunks
            # For now, we'll continue and just log the error
    
    if not embeddings:
        raise Exception("Failed to generate any embeddings")
        
    return embeddings