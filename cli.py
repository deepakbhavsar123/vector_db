#!/usr/bin/env python3
"""
Simple CLI interface for the Document Store
"""

import argparse
import sys
from pathlib import Path
from document_store import DocumentStore

def main():
    parser = argparse.ArgumentParser(description="Document Store CLI")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Store command
    store_parser = subparsers.add_parser('store', help='Store documents')
    store_parser.add_argument('--file', type=str, help='Store specific file')
    store_parser.add_argument('--all', action='store_true', help='Store all files in Files directory')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query documents')
    query_parser.add_argument('question', type=str, help='Question to ask')
    query_parser.add_argument('--model', type=str, default='llama3.2', help='Model to use')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search documents')
    search_parser.add_argument('query', type=str, help='Search query')
    search_parser.add_argument('--results', type=int, default=5, help='Number of results')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show database statistics')
    
    # Interactive command
    interactive_parser = subparsers.add_parser('interactive', help='Start interactive mode')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize document store
    doc_store = DocumentStore()
    
    if args.command == 'store':
        if args.file:
            success = doc_store.store_file(args.file)
            print(f"File storage {'successful' if success else 'failed'}")
        elif args.all:
            doc_store.store_all_files()
        else:
            print("Please specify --file <path> or --all")
    
    elif args.command == 'query':
        response = doc_store.query_with_llm(args.question, args.model)
        print(f"Question: {args.question}")
        print(f"Response: {response}")
    
    elif args.command == 'search':
        results = doc_store.search_documents(args.query, args.results)
        print(f"Search results for: {args.query}")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. From {result['metadata']['file_name']}:")
            print(f"   {result['document'][:200]}...")
            if result['distance']:
                print(f"   Similarity: {1 - result['distance']:.3f}")
    
    elif args.command == 'stats':
        stats = doc_store.get_database_stats()
        print("Database Statistics:")
        print(f"  Total files: {stats.get('total_files', 0)}")
        print(f"  Total chunks: {stats.get('total_chunks', 0)}")
        print(f"  Ollama available: {stats.get('ollama_available', False)}")
        print(f"  Available models: {', '.join(stats.get('available_models', []))}")
        print(f"  Files in database:")
        for file_path in stats.get('files', []):
            print(f"    - {Path(file_path).name}")
    
    elif args.command == 'interactive':
        print("\n" + "="*50)
        print("Document Query System - Interactive Mode")
        print("Commands:")
        print("  ask <question>    - Ask a question")
        print("  search <query>    - Search documents")
        print("  stats            - Show statistics")
        print("  quit             - Exit")
        print("="*50)
        
        while True:
            try:
                user_input = input("\n> ").strip()
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                if user_input.startswith('ask '):
                    question = user_input[4:].strip()
                    response = doc_store.query_with_llm(question)
                    print(f"\nResponse: {response}")
                
                elif user_input.startswith('search '):
                    query = user_input[7:].strip()
                    results = doc_store.search_documents(query, 3)
                    print(f"\nSearch results for: {query}")
                    for i, result in enumerate(results, 1):
                        print(f"\n{i}. From {result['metadata']['file_name']}:")
                        print(f"   {result['document'][:200]}...")
                
                elif user_input == 'stats':
                    stats = doc_store.get_database_stats()
                    print(f"\nFiles: {stats.get('total_files', 0)}")
                    print(f"Chunks: {stats.get('total_chunks', 0)}")
                    print(f"Ollama: {'Available' if stats.get('ollama_available') else 'Not available'}")
                
                else:
                    # Default to asking question
                    response = doc_store.query_with_llm(user_input)
                    print(f"\nResponse: {response}")
                    
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    main()
