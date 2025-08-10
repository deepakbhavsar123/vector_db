import os
import json
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from datetime import datetime

# Document processing libraries
from docx import Document
import PyPDF2
import pandas as pd

# Vector database and embeddings
import chromadb
from chromadb.config import Settings

# Ollama integration
import requests

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OllamaClient:
    """Client for interacting with Ollama API"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
    
    def is_available(self) -> bool:
        """Check if Ollama is running and available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        try:
            response = requests.get(f"{self.api_url}/tags")
            if response.status_code == 200:
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
            return []
        except requests.RequestException:
            return []
    
    def generate_embeddings(self, text: str, model: str = "nomic-embed-text") -> Optional[List[float]]:
        """Generate embeddings for text using Ollama"""
        try:
            payload = {
                "model": model,
                "prompt": text
            }
            response = requests.post(f"{self.api_url}/embeddings", json=payload, timeout=30)
            if response.status_code == 200:
                return response.json().get('embedding')
            else:
                logger.error(f"Failed to generate embeddings: {response.text}")
                return None
        except requests.RequestException as e:
            logger.error(f"Error generating embeddings: {e}")
            return None
    
    def chat(self, prompt: str, model: str = "llama3.2", context: str = "") -> str:
        """Chat with Ollama model"""
        try:
            full_prompt = f"Context: {context}\n\nQuestion: {prompt}" if context else prompt
            
            payload = {
                "model": model,
                "prompt": full_prompt,
                "stream": False
            }
            
            response = requests.post(f"{self.api_url}/generate", json=payload, timeout=60)
            if response.status_code == 200:
                return response.json().get('response', 'No response generated')
            else:
                logger.error(f"Failed to get response: {response.text}")
                return "Error: Failed to get response from Ollama"
        except requests.RequestException as e:
            logger.error(f"Error querying Ollama: {e}")
            return f"Error: {e}"

class DocumentProcessor:
    """Process different types of documents"""
    
    @staticmethod
    def extract_text_from_docx(file_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            doc = Document(file_path)
            text = []
            for paragraph in doc.paragraphs:
                text.append(paragraph.text)
            return '\n'.join(text)
        except Exception as e:
            logger.error(f"Error extracting text from DOCX {file_path}: {e}")
            return ""
    
    @staticmethod
    def extract_text_from_pdf(file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            text = []
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text.append(page.extract_text())
            return '\n'.join(text)
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {e}")
            return ""
    
    @staticmethod
    def extract_text_from_txt(file_path: str) -> str:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error extracting text from TXT {file_path}: {e}")
            return ""
    
    @staticmethod
    def extract_text_from_csv(file_path: str) -> str:
        """Extract text from CSV file"""
        try:
            df = pd.read_csv(file_path)
            return df.to_string()
        except Exception as e:
            logger.error(f"Error extracting text from CSV {file_path}: {e}")
            return ""
    
    @classmethod
    def extract_text(cls, file_path: str) -> str:
        """Extract text based on file extension"""
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.docx':
            return cls.extract_text_from_docx(file_path)
        elif file_ext == '.pdf':
            return cls.extract_text_from_pdf(file_path)
        elif file_ext == '.txt':
            return cls.extract_text_from_txt(file_path)
        elif file_ext == '.csv':
            return cls.extract_text_from_csv(file_path)
        else:
            logger.warning(f"Unsupported file type: {file_ext}")
            return ""

class DocumentStore:
    """Main document storage and querying system"""
    
    def __init__(self, files_directory: str = "Files", db_path: str = "chroma_db"):
        self.files_directory = Path(files_directory)
        self.db_path = db_path
        self.ollama_client = OllamaClient()
        self.processor = DocumentProcessor()
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Check Ollama availability
        if not self.ollama_client.is_available():
            logger.warning("Ollama is not available. Please make sure Ollama is running.")
            logger.info("Install Ollama from: https://ollama.ai/")
            logger.info("Then run: ollama pull nomic-embed-text && ollama pull llama3.2")
    
    def get_file_hash(self, file_path: str) -> str:
        """Generate hash for file to detect changes"""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary if possible
            if end < len(text):
                last_period = chunk.rfind('.')
                if last_period > chunk_size * 0.7:  # Only if period is in last 30%
                    chunk = chunk[:last_period + 1]
                    end = start + last_period + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
            
            if start >= len(text):
                break
        
        return chunks
    
    def store_file(self, file_path: str) -> bool:
        """Store a single file in the vector database"""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return False
            
            # Extract text
            text = self.processor.extract_text(str(file_path))
            if not text.strip():
                logger.warning(f"No text extracted from {file_path}")
                return False
            
            # Generate file metadata
            file_hash = self.get_file_hash(str(file_path))
            file_id = f"{file_path.stem}_{file_hash[:8]}"
            
            # Check if file already exists in database
            existing = self.collection.get(where={"file_path": str(file_path)})
            if existing['ids'] and existing['metadatas'][0].get('file_hash') == file_hash:
                logger.info(f"File {file_path.name} already up to date in database")
                return True
            
            # Delete existing entries for this file
            if existing['ids']:
                self.collection.delete(where={"file_path": str(file_path)})
            
            # Split text into chunks
            chunks = self.chunk_text(text)
            logger.info(f"Split {file_path.name} into {len(chunks)} chunks")
            
            # Generate embeddings and store chunks
            stored_count = 0
            for i, chunk in enumerate(chunks):
                chunk_id = f"{file_id}_chunk_{i}"
                
                # Use ChromaDB's built-in embeddings if Ollama is not available
                if self.ollama_client.is_available():
                    embedding = self.ollama_client.generate_embeddings(chunk)
                    if embedding is None:
                        logger.warning(f"Failed to generate embedding for chunk {i}")
                        continue
                else:
                    embedding = None  # ChromaDB will generate embeddings
                
                metadata = {
                    "file_path": str(file_path),
                    "file_name": file_path.name,
                    "file_hash": file_hash,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "processed_at": datetime.now().isoformat()
                }
                
                try:
                    if embedding:
                        self.collection.add(
                            ids=[chunk_id],
                            documents=[chunk],
                            metadatas=[metadata],
                            embeddings=[embedding]
                        )
                    else:
                        self.collection.add(
                            ids=[chunk_id],
                            documents=[chunk],
                            metadatas=[metadata]
                        )
                    stored_count += 1
                except Exception as e:
                    logger.error(f"Error storing chunk {i}: {e}")
            
            logger.info(f"Successfully stored {stored_count}/{len(chunks)} chunks for {file_path.name}")
            return stored_count > 0
            
        except Exception as e:
            logger.error(f"Error storing file {file_path}: {e}")
            return False
    
    def store_all_files(self) -> None:
        """Store all files from the Files directory"""
        if not self.files_directory.exists():
            logger.error(f"Files directory not found: {self.files_directory}")
            return
        
        supported_extensions = {'.docx', '.pdf', '.txt', '.csv'}
        files_processed = 0
        
        for file_path in self.files_directory.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                logger.info(f"Processing file: {file_path.name}")
                if self.store_file(str(file_path)):
                    files_processed += 1
        
        logger.info(f"Processed {files_processed} files")
    
    def search_documents(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents"""
        try:
            # Generate embedding for query if Ollama is available
            if self.ollama_client.is_available():
                query_embedding = self.ollama_client.generate_embeddings(query)
                if query_embedding:
                    results = self.collection.query(
                        query_embeddings=[query_embedding],
                        n_results=n_results
                    )
                else:
                    # Fallback to text search
                    results = self.collection.query(
                        query_texts=[query],
                        n_results=n_results
                    )
            else:
                # Use ChromaDB's built-in embedding
                results = self.collection.query(
                    query_texts=[query],
                    n_results=n_results
                )
            
            # Format results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    'id': results['ids'][0][i],
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                })
            
            return formatted_results
        
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def query_with_llm(self, question: str, model: str = "llama3.2") -> str:
        """Query documents and get LLM response"""
        if not self.ollama_client.is_available():
            return "Error: Ollama is not available. Please install and run Ollama."
        
        # Search for relevant documents
        search_results = self.search_documents(question, n_results=3)
        
        if not search_results:
            return "No relevant documents found."
        
        # Build context from search results
        context_parts = []
        for result in search_results:
            file_name = result['metadata']['file_name']
            content = result['document']
            context_parts.append(f"From {file_name}:\n{content}")
        
        context = "\n\n".join(context_parts)
        
        # Get response from LLM
        return self.ollama_client.chat(question, model, context)
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the database"""
        try:
            all_data = self.collection.get()
            total_chunks = len(all_data['ids'])
            
            # Count files
            file_paths = set()
            for metadata in all_data['metadatas']:
                file_paths.add(metadata['file_path'])
            
            return {
                'total_files': len(file_paths),
                'total_chunks': total_chunks,
                'files': list(file_paths),
                'ollama_available': self.ollama_client.is_available(),
                'available_models': self.ollama_client.get_available_models()
            }
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}

def main():
    """Example usage"""
    # Initialize document store
    doc_store = DocumentStore()
    
    # Store all files
    print("Storing files...")
    doc_store.store_all_files()
    
    # Show database stats
    stats = doc_store.get_database_stats()
    print(f"\nDatabase Stats:")
    print(f"Files: {stats.get('total_files', 0)}")
    print(f"Chunks: {stats.get('total_chunks', 0)}")
    print(f"Ollama available: {stats.get('ollama_available', False)}")
    
    # Interactive query loop
    print("\n" + "="*50)
    print("Document Query System")
    print("Type 'quit' to exit")
    print("="*50)
    
    while True:
        question = input("\nEnter your question: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        if not question:
            continue
        
        print("\nSearching documents...")
        response = doc_store.query_with_llm(question)
        print(f"\nResponse:\n{response}")

if __name__ == "__main__":
    main()
