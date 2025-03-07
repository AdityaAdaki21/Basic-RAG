import os
import requests
import json
import PyPDF2
import numpy as np
from typing import List, Dict, Any, Tuple
import argparse
import concurrent.futures
import time
import sys

class OllamaRAG:
    def __init__(self, llm_model="llama3.2", embedding_model="mxbai-embed-large"):
        """
        Initialize the RAG system with specified models.
        
        Args:
            llm_model: The Ollama LLM model to use for generation
            embedding_model: The Ollama embedding model to use
        """
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.documents = []
        self.embeddings = []
        self.base_url = "http://localhost:11434/api"
        self.current_pdf = None
        
        # Verify that the models are available in Ollama
        self._verify_models()
    
    def _verify_models(self) -> None:
        """Verify that the required models are available."""
        try:
            response = requests.get(f"{self.base_url}/tags")
            available_models = [model["name"] for model in response.json()["models"]]
            
            missing_models = []
            if self.llm_model not in available_models:
                missing_models.append(self.llm_model)
            if self.embedding_model not in available_models:
                missing_models.append(self.embedding_model)
            
            if missing_models:
                print(f"Warning: The following models are not available: {', '.join(missing_models)}")
                print("Please pull these models using: ollama pull <model_name>")
                print("Command examples:")
                for model in missing_models:
                    print(f"  ollama pull {model}")
        except Exception as e:
            print(f"Error connecting to Ollama server: {e}")
            print("Make sure the Ollama server is running with 'ollama serve'")
            sys.exit(1)
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embeddings for a piece of text.
        
        Args:
            text: The text to embed
            
        Returns:
            List of embedding values
        """
        response = requests.post(
            f"{self.base_url}/embeddings",
            json={"model": self.embedding_model, "prompt": text}
        )
        
        if response.status_code != 200:
            raise Exception(f"Error getting embeddings: {response.text}")
        
        return response.json()["embedding"]
    
    def _process_chunk(self, chunk_info: Dict) -> Dict:
        """Process a single chunk with embedding (for parallel processing)"""
        chunk, pdf_path, chunk_num, total_chunks = chunk_info["chunk"], chunk_info["pdf_path"], chunk_info["chunk_num"], chunk_info["total_chunks"]
        
        try:
            embedding = self.get_embedding(chunk)
            sys.stdout.write(f"\rProcessed chunk {chunk_num}/{total_chunks} ({int(chunk_num/total_chunks*100)}%)")
            sys.stdout.flush()
            return {"text": chunk, "source": pdf_path, "embedding": embedding, "success": True}
        except Exception as e:
            return {"text": chunk, "source": pdf_path, "embedding": None, "success": False, "error": str(e)}
    
    def ingest_pdf(self, pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200, 
                   max_workers: int = 4) -> None:
        """
        Ingest a PDF document, chunk it, and compute embeddings using parallel processing.
        
        Args:
            pdf_path: Path to the PDF file
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks in characters
            max_workers: Maximum number of parallel workers for embedding computation
        """
        self.current_pdf = os.path.basename(pdf_path)
        print(f"Ingesting PDF: {pdf_path}")
        start_time = time.time()
        
        # Extract text from PDF
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                total_pages = len(reader.pages)
                
                for i, page in enumerate(reader.pages):
                    sys.stdout.write(f"\rExtracting text: page {i+1}/{total_pages} ({int((i+1)/total_pages*100)}%)")
                    sys.stdout.flush()
                    text += page.extract_text() + " "
                print("\nText extraction complete.")
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return
        
        # Create chunks with overlap
        chunks = []
        for i in range(0, len(text), chunk_size - chunk_overlap):
            chunk = text[i:i + chunk_size]
            if len(chunk) > 50:  # Only keep chunks with meaningful content
                chunks.append(chunk)
        
        print(f"Created {len(chunks)} chunks from the PDF")
        
        # Prepare chunk information for parallel processing
        chunk_infos = [
            {"chunk": chunk, "pdf_path": pdf_path, "chunk_num": i+1, "total_chunks": len(chunks)}
            for i, chunk in enumerate(chunks)
        ]
        
        # Process chunks in parallel
        print("Computing embeddings in parallel...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(self._process_chunk, chunk_infos))
        
        # Process results
        successful_chunks = 0
        for result in results:
            if result["success"]:
                self.documents.append({"text": result["text"], "source": result["source"]})
                self.embeddings.append(result["embedding"])
                successful_chunks += 1
        
        print(f"\nSuccessfully processed {successful_chunks}/{len(chunks)} chunks with embeddings")
        print(f"Ingestion completed in {time.time() - start_time:.2f} seconds")
    
    def find_similar_chunks(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Find chunks most similar to the query.
        
        Args:
            query: The query text
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries containing similar chunks and their similarity scores
        """
        if not self.embeddings:
            return []
            
        query_embedding = self.get_embedding(query)
        
        # Calculate cosine similarity
        def cosine_similarity(a, b):
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            return np.dot(a, b) / (norm_a * norm_b)
        
        similarities = [
            cosine_similarity(query_embedding, doc_embedding)
            for doc_embedding in self.embeddings
        ]
        
        # Get indices of top_k most similar chunks
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                "text": self.documents[idx]["text"],
                "source": self.documents[idx]["source"],
                "similarity": similarities[idx]
            })
        
        return results
    
    def generate_response(self, query: str, system_prompt: str = None, top_k: int = 3) -> str:
        """
        Generate a response to the query using the LLM and retrieved context.
        
        Args:
            query: The user query
            system_prompt: Optional system prompt to guide the LLM
            top_k: Number of relevant chunks to include in context
            
        Returns:
            The generated response
        """
        # Find relevant chunks
        relevant_chunks = self.find_similar_chunks(query, top_k=top_k)
        
        if not relevant_chunks:
            return "No relevant information found. Please ingest a PDF document first."
        
        # Create context from retrieved chunks
        context = "\n\n".join([chunk["text"] for chunk in relevant_chunks])
        
        # Default system prompt if none provided
        if not system_prompt:
            system_prompt = (
                "You are a helpful assistant. Answer the user's question based on the provided context. "
                "If the context doesn't contain relevant information, say you don't know."
            )
        
        # Prepare the prompt with context and query
        prompt = f"""Context information is below.
---------------------
{context}
---------------------

Given the context information and not prior knowledge, answer the following question:
{query}
"""
        
        print("Generating response...")
        # Send the request to Ollama
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "model": self.llm_model,
                "prompt": prompt,
                "system": system_prompt,
                "stream": False
            }
        )
        
        if response.status_code != 200:
            return f"Error generating response: {response.text}"
        
        return response.json()["response"]

    def save_knowledge_base(self, filepath: str = None) -> None:
        """
        Save the current knowledge base to a file.
        
        Args:
            filepath: Path to save the knowledge base. If None, use PDF name.
        """
        if not filepath and self.current_pdf:
            filepath = f"{os.path.splitext(self.current_pdf)[0]}.kb.json"
        elif not filepath:
            filepath = "knowledge_base.json"
            
        data = {
            "documents": self.documents,
            "embeddings": self.embeddings,
            "pdf_name": self.current_pdf
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f)
        
        print(f"Knowledge base saved to {filepath}")
    
    def load_knowledge_base(self, filepath: str) -> None:
        """
        Load a knowledge base from a file.
        
        Args:
            filepath: Path to the knowledge base file
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.documents = data["documents"]
        self.embeddings = data["embeddings"]
        self.current_pdf = data.get("pdf_name", os.path.basename(filepath))
        
        print(f"Loaded knowledge base with {len(self.documents)} documents")


def main():
    parser = argparse.ArgumentParser(description="PDF-based RAG using Ollama")
    parser.add_argument("--pdf", type=str, help="Path to PDF file to ingest")
    parser.add_argument("--kb", type=str, help="Path to knowledge base file to load")
    parser.add_argument("--llm", type=str, default="llama3.2", help="LLM model to use")
    parser.add_argument("--embedder", type=str, default="mxbai-embed-large", help="Embedding model to use")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers for embedding")
    
    args = parser.parse_args()
    
    # Initialize the RAG system
    rag = OllamaRAG(llm_model=args.llm, embedding_model=args.embedder)
    
    # Initial setup - either load PDF or knowledge base
    if args.pdf:
        rag.ingest_pdf(args.pdf, max_workers=args.workers)
        # Auto-save knowledge base after ingestion
        rag.save_knowledge_base()
    elif args.kb:
        rag.load_knowledge_base(args.kb)
    
    # Interactive mode for asking questions
    pdf_info = f" about {rag.current_pdf}" if rag.current_pdf else ""
    if rag.documents:
        print(f"\nPDF RAG System - Ready to answer questions{pdf_info}")
        print("==================================")
        print("Type 'exit' to quit, 'save' to save knowledge base, 'help' for commands")
        
        while True:
            user_input = input("\nEnter your question: ")
            
            if user_input.lower() == 'exit':
                break
            elif user_input.lower() == 'save':
                rag.save_knowledge_base()
            elif user_input.lower() == 'help':
                print("\nAvailable commands:")
                print("  exit - Exit the program")
                print("  save - Save the current knowledge base")
                print("  help - Show this help message")
                print("  Any other input will be treated as a question to answer")
            else:
                response = rag.generate_response(user_input)
                print("\nResponse:")
                print(response)
    else:
        print("\nNo documents loaded. Please provide a PDF file or knowledge base.")
        print("Example usage:")
        print("  python rag_system.py --pdf document.pdf")
        print("  python rag_system.py --kb knowledge_base.json")

if __name__ == "__main__":
    main()