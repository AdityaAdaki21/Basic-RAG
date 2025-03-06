# document_loader.py

import os
import PyPDF2
from typing import List, Dict, Optional, Union, Any
import logging
from docx import Document
import csv
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentLoader:
    """Handles loading and processing of various document types."""
    
    def __init__(self, chunk_size: int = 150, chunk_overlap: int = 20):
        """
        Initialize the document loader.
        
        Args:
            chunk_size: Number of words per chunk
            chunk_overlap: Number of overlapping words between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.supported_extensions = {
            '.pdf': self._load_pdf,
            '.txt': self._load_text,
            '.docx': self._load_docx,
            '.csv': self._load_csv,
            '.json': self._load_json
        }
    
    def load_document(self, file_path: str) -> str:
        """
        Load text from a document file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Extracted text content
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        _, ext = os.path.splitext(file_path.lower())
        
        if ext not in self.supported_extensions:
            supported = ", ".join(self.supported_extensions.keys())
            raise ValueError(f"Unsupported file type: {ext}. Supported types: {supported}")
            
        try:
            return self.supported_extensions[ext](file_path)
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {e}")
            raise Exception(f"Failed to load document: {e}")
    
    def _load_pdf(self, file_path: str) -> str:
        """Load text from a PDF file."""
        text = ""
        try:
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise
    
    def _load_text(self, file_path: str) -> str:
        """Load text from a plain text file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            # Try different encodings if UTF-8 fails
            with open(file_path, "r", encoding="latin-1") as f:
                return f.read()
    
    def _load_docx(self, file_path: str) -> str:
        """Load text from a Word document."""
        try:
            doc = Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            logger.error(f"Error extracting text from DOCX: {e}")
            raise
        
    def _load_csv(self, file_path: str) -> str:
        """Load text from a CSV file."""
        text = ""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    text += " | ".join(row) + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting text from CSV: {e}")
            raise
    
    def _load_json(self, file_path: str) -> str:
        """Load text from a JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return json.dumps(data, indent=2)
        except Exception as e:
            logger.error(f"Error extracting text from JSON: {e}")
            raise
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        if not text.strip():
            return []
            
        words = text.split()
        chunks = []
        
        # Use sliding window with overlap
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = " ".join(words[i:i + self.chunk_size])
            if chunk:
                chunks.append(chunk)
                
        return chunks
    
    def process_file(self, file_path: str) -> List[str]:
        """
        Load and split a document file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of text chunks
        """
        text = self.load_document(file_path)
        return self.split_text(text)

# For backward compatibility
def load_pdf_text(pdf_path):
    """Legacy function for backward compatibility."""
    loader = DocumentLoader()
    return loader.load_document(pdf_path)

def split_text(text, chunk_size=150):
    """Legacy function for backward compatibility."""
    loader = DocumentLoader(chunk_size=chunk_size)
    return loader.split_text(text)