# main.py

import sys
import numpy as np
import PyPDF2
from embedding import compute_embeddings_ollama, ollama_embed
from db import build_index, search_index
from llm import call_ollama

def load_pdf_text(pdf_path):
    """
    Extract text from the PDF file.
    """
    text = ""
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        return text
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return ""

def split_text(text, chunk_size=150):  # Reduced chunk size
    """
    Split the text into chunks based on word count.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def main(pdf_path):
    # 1. Load PDF text and split it into chunks.
    print("Extracting text from PDF...")
    text = load_pdf_text(pdf_path)
    if not text:
        print("Failed to extract text from PDF. Exiting.")
        return
        
    chunks = split_text(text)
    print(f"Total chunks created: {len(chunks)}")
    
    # 2. Compute embeddings for chunks using Ollama.
    print("Computing embeddings for chunks using Ollama...")
    try:
        embeddings = compute_embeddings_ollama(chunks)
        embeddings = np.array(embeddings).astype("float32")
    except Exception as e:
        print(f"Error computing embeddings: {e}")
        return
    
    # 3. Build the FAISS index.
    print("Building FAISS index...")
    index = build_index(embeddings)
    
    # 4. Interactive query loop.
    print("\nReady for queries. Type 'exit' to quit.")
    while True:
        query = input("\nEnter your query: ")
        if query.lower() == "exit":
            break
        
        try:
            # Compute query embedding using Ollama.
            print("Computing query embedding...")
            query_embedding = ollama_embed(query)
            query_embedding = np.array(query_embedding).astype("float32").reshape(1, -1)
            
            # Retrieve the top 3 most similar text chunks.
            print("Searching for relevant information...")
            distances, indices = search_index(query_embedding, index, top_k=3)
            retrieved_chunks = [chunks[i] for i in indices[0]]
            
            # Display similarity scores
            print("\nRetrieved passages (similarity scores):")
            for i, (chunk, dist) in enumerate(zip(retrieved_chunks, distances[0])):
                similarity = 1 / (1 + dist)  # Convert distance to similarity
                print(f"Passage {i+1} (similarity: {similarity:.4f})")
            
            context = "\n\n".join(retrieved_chunks)
            
            # Build the prompt.
            prompt = f"""Context information is below.
---------------------
{context}
---------------------
Given the context information and not prior knowledge, answer the question: {query}
"""
            
            # Generate an answer using Ollama.
            print("Generating answer...")
            response = call_ollama(prompt, model="llama3.2")
            print("\nAnswer:")
            print(response)
            print("-" * 50)
        except Exception as e:
            print(f"Error processing query: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <pdf_file>")
    else:
        main(sys.argv[1])