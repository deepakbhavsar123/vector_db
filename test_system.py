#!/usr/bin/env python3
"""
Quick test script to verify the basic functionality
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    
    try:
        from docx import Document
        print("✅ python-docx imported successfully")
    except ImportError:
        print("❌ python-docx not found. Install with: pip install python-docx")
        return False
    
    try:
        import PyPDF2
        print("✅ PyPDF2 imported successfully")
    except ImportError:
        print("❌ PyPDF2 not found. Install with: pip install PyPDF2")
        return False
    
    try:
        import pandas as pd
        print("✅ pandas imported successfully")
    except ImportError:
        print("❌ pandas not found. Install with: pip install pandas")
        return False
    
    try:
        import chromadb
        print("✅ chromadb imported successfully")
    except ImportError:
        print("❌ chromadb not found. Install with: pip install chromadb")
        return False
    
    try:
        import requests
        print("✅ requests imported successfully")
    except ImportError:
        print("❌ requests not found. Install with: pip install requests")
        return False
    
    return True

def test_file_processing():
    """Test document processing without Ollama"""
    print("\nTesting document processing...")
    
    try:
        from document_store import DocumentProcessor
        processor = DocumentProcessor()
        
        # Test with the existing DOCX file
        files_dir = Path("Files")
        if files_dir.exists():
            docx_files = list(files_dir.glob("*.docx"))
            if docx_files:
                test_file = docx_files[0]
                print(f"Testing with file: {test_file.name}")
                
                text = processor.extract_text(str(test_file))
                if text.strip():
                    print(f"✅ Successfully extracted {len(text)} characters")
                    print(f"   Preview: {text[:100]}...")
                    return True
                else:
                    print("❌ No text extracted from file")
                    return False
            else:
                print("❌ No DOCX files found in Files directory")
                return False
        else:
            print("❌ Files directory not found")
            return False
            
    except Exception as e:
        print(f"❌ Error during file processing: {e}")
        return False

def test_vector_db():
    """Test ChromaDB functionality"""
    print("\nTesting vector database...")
    
    try:
        import chromadb
        from chromadb.config import Settings
        
        # Create a temporary client
        client = chromadb.Client()
        collection = client.create_collection("test_collection")
        
        # Add a test document
        collection.add(
            documents=["This is a test document"],
            metadatas=[{"source": "test"}],
            ids=["test_1"]
        )
        
        # Query the document
        results = collection.query(
            query_texts=["test"],
            n_results=1
        )
        
        if results['documents'][0][0] == "This is a test document":
            print("✅ ChromaDB working correctly")
            return True
        else:
            print("❌ ChromaDB query failed")
            return False
            
    except Exception as e:
        print(f"❌ Error with ChromaDB: {e}")
        return False

def test_ollama_connection():
    """Test Ollama connection"""
    print("\nTesting Ollama connection...")
    
    try:
        from document_store import OllamaClient
        ollama = OllamaClient()
        
        if ollama.is_available():
            models = ollama.get_available_models()
            print(f"✅ Ollama is running with models: {models}")
            
            # Check for required models
            required_models = ['nomic-embed-text', 'llama3.2']
            missing_models = []
            
            for model in required_models:
                if not any(model in m for m in models):
                    missing_models.append(model)
            
            if missing_models:
                print(f"⚠️  Missing required models: {missing_models}")
                print("   Install with: " + " && ".join([f"ollama pull {m}" for m in missing_models]))
                return False
            else:
                print("✅ All required models are available")
                return True
        else:
            print("❌ Ollama is not running")
            print("   Start with: ollama serve")
            print("   Install models with: ollama pull nomic-embed-text && ollama pull llama3.2")
            return False
            
    except Exception as e:
        print(f"❌ Error connecting to Ollama: {e}")
        return False

def main():
    print("🔍 Document Store - System Check")
    print("================================")
    
    all_tests_passed = True
    
    # Test imports
    if not test_imports():
        all_tests_passed = False
    
    # Test file processing
    if not test_file_processing():
        all_tests_passed = False
    
    # Test vector database
    if not test_vector_db():
        all_tests_passed = False
    
    # Test Ollama (optional)
    ollama_works = test_ollama_connection()
    
    print("\n" + "="*40)
    if all_tests_passed:
        print("✅ Core functionality tests passed!")
        if ollama_works:
            print("✅ Ollama integration ready!")
            print("🚀 You can run: python demo.py")
        else:
            print("⚠️  Ollama not ready, but basic functionality works")
            print("📝 You can still process and search documents")
    else:
        print("❌ Some tests failed. Please install missing dependencies.")
        print("💡 Run: pip install -r requirements.txt")
    
    print("\n📋 Next steps:")
    if not ollama_works:
        print("   1. Install Ollama: curl -fsSL https://ollama.ai/install.sh | sh")
        print("   2. Start Ollama: ollama serve")
        print("   3. Install models: ollama pull nomic-embed-text && ollama pull llama3.2")
    print("   4. Run the demo: python demo.py")
    print("   5. Use CLI: python cli.py interactive")

if __name__ == "__main__":
    main()
