# main.py

import sys
import os
import numpy as np
import argparse
import logging
import time
from colorama import init, Fore, Style
from typing import List, Dict, Optional

# Import our modules
from embedding import EmbeddingClient
from llm import LLMClient
from db import VectorStore
from document_loader import DocumentLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rag_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize colorama for colored terminal output
init()

class RAGSystem:
    """Retrieval-Augmented Generation system for document querying."""
    
    def __init__(self, 
                embedding_model: str = "mxbai-embed-large", 
                llm_model: str = "llama3.2",
                chunk_size: int = 150,
                chunk_overlap: int = 20,
                ollama_base_url: str = "http://localhost:11434"):
        """
        Initialize the RAG system.
        
        Args:
            embedding_model: Model to use for embeddings
            llm_model: Model to use for text generation
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            ollama_base_url: Base URL for Ollama API
        """
        self.embedding_client = EmbeddingClient(base_url=ollama_base_url, default_model=embedding_model)
        self.llm_client = LLMClient(base_url=ollama_base_url, default_model=llm_model)
        self.document_loader = DocumentLoader(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.vector_store = None
        self.chunks = []
        
        self.config = {
            "embedding_model": embedding_model,
            "llm_model": llm_model,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "ollama_base_url": ollama_base_url
        }
        
        logger.info(f"RAG System initialized with config: {self.config}")
    
    def process_document(self, file_path: str) -> None:
        """
        Process a document file: load, chunk, embed, and index.
        
        Args:
            file_path: Path to the document file
        """
        start_time = time.time()
        
        # 1. Load and split the document
        print(f"{Fore.CYAN}Loading document: {file_path}{Style.RESET_ALL}")
        self.chunks = self.document_loader.process_file(file_path)
        print(f"{Fore.GREEN}Created {len(self.chunks)} chunks{Style.RESET_ALL}")
        
        # 2. Generate embeddings
        print(f"{Fore.CYAN}Generating embeddings...{Style.RESET_ALL}")
        
        def progress_callback(current, total):
            percent = (current + 1) / total * 100
            bar_length = 30
            filled_length = int(bar_length * percent / 100)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            print(f"\r{Fore.YELLOW}Progress: [{bar}] {percent:.1f}% ({current+1}/{total}){Style.RESET_ALL}", end="")
            
        embeddings = self.embedding_client.compute_batch_embeddings(
            self.chunks, 
            progress_callback=progress_callback
        )
        print("\n")  # Finish the progress bar line
        
        # 3. Build the vector store
        print(f"{Fore.CYAN}Building vector index...{Style.RESET_ALL}")
        embeddings_array = np.array(embeddings).astype("float32")
        
        self.vector_store = VectorStore()
        self.vector_store.create_index(embeddings_array.shape[1])
        self.vector_store.add_embeddings(embeddings_array, self.chunks)
        
        elapsed_time = time.time() - start_time
        print(f"{Fore.GREEN}Document processing complete in {elapsed_time:.2f} seconds{Style.RESET_ALL}")
    
    def save_index(self, directory: str = "saved_index") -> None:
        """
        Save the current vector index and metadata.
        
        Args:
            directory: Directory to save to
        """
        if not self.vector_store:
            print(f"{Fore.RED}No index to save{Style.RESET_ALL}")
            return
            
        path = self.vector_store.save(directory)
        print(f"{Fore.GREEN}Index saved to: {path}{Style.RESET_ALL}")
    
    def load_index(self, directory: str = "saved_index") -> None:
        """
        Load a saved vector index and metadata.
        
        Args:
            directory: Directory to load from
        """
        try:
            self.vector_store = VectorStore()
            self.vector_store.load(directory)
            self.chunks = self.vector_store.metadata.get("chunks", [])
            print(f"{Fore.GREEN}Loaded index with {self.vector_store.metadata['total_vectors']} vectors{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error loading index: {e}{Style.RESET_ALL}")
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: User query
            top_k: Number of results to retrieve
            
        Returns:
            List of retrieved chunks with metadata
        """
        if not self.vector_store:
            raise ValueError("No vector store available. Process a document first.")
            
        # 1. Generate query embedding
        query_embedding = self.embedding_client.embed(query)
        query_embedding = np.array(query_embedding).astype("float32").reshape(1, -1)
        
        # 2. Search the index
        distances, indices = self.vector_store.search(query_embedding, top_k=top_k)
        
        # 3. Prepare results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                similarity = 1 / (1 + distances[0][i])  # Convert distance to similarity
                results.append({
                    "chunk": self.chunks[idx],
                    "index": idx,
                    "similarity": similarity
                })
                
        return results
    
    def answer_query(self, query: str, top_k: int = 3) -> str:
        """
        Answer a query using RAG.
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve
            
        Returns:
            Generated answer
        """
        # 1. Retrieve relevant chunks
        try:
            results = self.retrieve(query, top_k=top_k)
        except Exception as e:
            logger.error(f"Error retrieving results: {e}")
            return f"Error: {e}"
        
        # 2. Display retrieved passages
        print(f"\n{Fore.CYAN}Retrieved passages:{Style.RESET_ALL}")
        for i, result in enumerate(results):
            print(f"{Fore.YELLOW}Passage {i+1} (similarity: {result['similarity']:.4f}){Style.RESET_ALL}")
            print(f"{result['chunk'][:100]}...")
        
        # 3. Build context from retrieved chunks
        context = "\n\n".join([r["chunk"] for r in results])
        
        # 4. Generate the answer
        print(f"\n{Fore.CYAN}Generating answer...{Style.RESET_ALL}")
        response = self.llm_client.generate_with_context(query, context)
        
        return response

def main():
    """Main function to run the RAG system."""
    parser = argparse.ArgumentParser(description="Retrieval-Augmented Generation System")
    parser.add_argument("--file", "-f", help="Path to the document file to process")
    parser.add_argument("--load", "-l", help="Load a saved index from the specified directory")
    parser.add_argument("--save", "-s", help="Save the index to the specified directory")
    parser.add_argument("--chunk-size", type=int, default=150, help="Size of text chunks")
    parser.add_argument("--chunk-overlap", type=int, default=20, help="Overlap between chunks")
    parser.add_argument("--embedding-model", default="mxbai-embed-large", help="Embedding model to use")
    parser.add_argument("--llm-model", default="llama3.2", help="LLM model to use")
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve")
    
    args = parser.parse_args()
    
    # Initialize the RAG system
    rag = RAGSystem(
        embedding_model=args.embedding_model,
        llm_model=args.llm_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    # Process file or load index
    if args.file:
        rag.process_document(args.file)
        if args.save:
            rag.save_index(args.save)
    elif args.load:
        rag.load_index(args.load)
    else:
        parser.print_help()
        return
    
    # Interactive query loop
    print(f"\n{Fore.GREEN}Ready for queries. Type 'exit' to quit, 'help' for commands.{Style.RESET_ALL}")
    
    while True:
        try:
            query = input(f"\n{Fore.CYAN}Enter your query: {Style.RESET_ALL}")
            
            if query.lower() == "exit":
                break
            elif query.lower() == "help":
                print(f"\n{Fore.YELLOW}Available commands:{Style.RESET_ALL}")
                print("  exit - Exit the program")
                print("  help - Show this help message")
                print("  save <dir> - Save the current index")
                print("  load <dir> - Load an index")
                print("  topk <number> - Set the number of passages to retrieve")
                continue
            elif query.lower().startswith("save "):
                dir_name = query[5:].strip()
                if dir_name:
                    rag.save_index(dir_name)
                continue
            elif query.lower().startswith("load "):
                dir_name = query[5:].strip()
                if dir_name:
                    rag.load_index(dir_name)
                continue
            elif query.lower().startswith("topk "):
                try:
                    top_k = int(query[5:].strip())
                    args.top_k = top_k
                    print(f"{Fore.GREEN}Set top-k to {top_k}{Style.RESET_ALL}")
                except ValueError:
                    print(f"{Fore.RED}Invalid value for top-k{Style.RESET_ALL}")
                continue
                
            # Answer the query
            try:
                answer = rag.answer_query(query, top_k=args.top_k)
                print(f"\n{Fore.GREEN}Answer:{Style.RESET_ALL}")
                print(answer)
                print("-" * 50)
            except Exception as e:
                logger.error(f"Error answering query: {e}")
                print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
                
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Interrupted. Type 'exit' to quit.{Style.RESET_ALL}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            print(f"{Fore.RED}Unexpected error: {e}{Style.RESET_ALL}")

if __name__ == "__main__":
    main()