# PDF Question Answering System

A Retrieval-Augmented Generation (RAG) system that allows you to ask questions about PDF documents and receive AI-generated answers based on the document content.

## Overview

This application extracts text from PDF documents, splits it into manageable chunks, creates vector embeddings for each chunk using Ollama's embedding models, and builds a searchable vector database using FAISS. When you ask a question, the system:

1. Converts your question into an embedding
2. Finds the most relevant text chunks from the PDF using similarity search
3. Uses a large language model to generate an answer based on those relevant chunks

## Features

- PDF text extraction using PyPDF2
- Vector embeddings generation with Ollama's embedding models
- Fast similarity search with FAISS
- Natural language responses using Ollama LLMs
- Interactive query interface

## Requirements

- Python 3.9
- Ollama running locally with the following models:
  - `mxbai-embed-large` (for embeddings)
  - `llama3.2` (for text generation)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/AdityaAdaki21/Basic-RAG.git
   cd BASIC-RAG
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Make sure Ollama is installed and running locally:
   ```
   # Check Ollama status and install models if needed
   ollama list
   ollama pull mxbai-embed-large
   ollama pull llama3.2
   ```

## Usage

1. Run the application with a PDF file:
   ```
    python main.py --file path/to/your/document.pdf --chunk-size 1024
   ```

2. When prompted, enter your questions about the PDF content. Type 'exit' to quit.

## How It Works

The system uses a Retrieval-Augmented Generation (RAG) architecture:

- **Document Processing:**
  - PDF text is extracted and split into chunks of approximately 150 words each
  - Embeddings are generated for each chunk using the `mxbai-embed-large` model

- **Query Processing:**
  - User questions are converted to embeddings using the same model
  - FAISS finds the most similar text chunks based on vector similarity
  - The top 3 most relevant chunks are used as context for the LLM

- **Answer Generation:**
  - The context and question are formatted into a prompt
  - The `llama3.2` model generates a natural language answer based on the provided context

## Project Structure

- `main.py` - The main application script
- `db.py` - FAISS index and search functionality
- `embedding.py` - Text embedding generation using Ollama
- `llm.py` - LLM integration for answer generation
- `requirements.txt` - Required Python packages

## Performance Notes

- The system displays similarity scores for transparency
- All processing happens locally, with no data sent to external services

## Troubleshooting

- Ensure Ollama is running on the default port (11434)
- If you encounter embedding errors, try breaking the content into smaller chunks
- For better results with technical documents, consider using domain-specific LLMs


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.