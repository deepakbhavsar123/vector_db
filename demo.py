#!/usr/bin/env python3
"""
Example script demonstrating the Document Store functionality
"""

from document_store import DocumentStore
import time

def demo():
    print("🚀 Document Store Demo")
    print("=====================")
    
    # Initialize the document store
    print("\n1. Initializing Document Store...")
    doc_store = DocumentStore()
    
    # Check if Ollama is available
    if not doc_store.ollama_client.is_available():
        print("⚠️  Ollama is not running. Please start it with: ollama serve")
        print("   Also make sure you have the required models:")
        print("   ollama pull nomic-embed-text")
        print("   ollama pull llama3.2")
        return
    
    print("✅ Ollama is available!")
    
    # Show available models
    models = doc_store.ollama_client.get_available_models()
    print(f"📦 Available models: {', '.join(models)}")
    
    # Process and store files
    print("\n2. Processing and storing files...")
    doc_store.store_all_files()
    
    # Show database statistics
    print("\n3. Database Statistics:")
    stats = doc_store.get_database_stats()
    print(f"   📁 Files processed: {stats.get('total_files', 0)}")
    print(f"   📄 Text chunks created: {stats.get('total_chunks', 0)}")
    
    # Example queries
    print("\n4. Example Queries:")
    print("-" * 40)
    
    example_questions = [
        "What is the candidate's name and current position?",
        "What programming languages and technologies are mentioned?",
        "Tell me about the work experience",
        "What education background does the candidate have?"
    ]
    
    for i, question in enumerate(example_questions, 1):
        print(f"\n🤔 Question {i}: {question}")
        print("🤖 Thinking...")
        
        response = doc_store.query_with_llm(question)
        print(f"💡 Answer: {response}")
        
        # Add small delay between queries
        time.sleep(1)
    
    print("\n" + "="*50)
    print("✨ Demo completed!")
    print("💡 You can now use:")
    print("   - python cli.py interactive  (for interactive mode)")
    print("   - python cli.py query 'your question'  (for single queries)")
    print("   - python document_store.py  (for full interactive session)")

if __name__ == "__main__":
    demo()
