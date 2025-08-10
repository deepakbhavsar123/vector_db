#!/bin/bash

echo "Setting up Document Store with Ollama Integration"
echo "================================================="

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo ""
    echo "Ollama is not installed. Please install it first:"
    echo "Visit: https://ollama.ai/"
    echo ""
    echo "For Linux:"
    echo "curl -fsSL https://ollama.ai/install.sh | sh"
    echo ""
    exit 1
fi

# Start Ollama service (if not running)
echo "Starting Ollama service..."
ollama serve &
sleep 5

# Pull required models
echo "Pulling required Ollama models..."
echo "This may take a while depending on your internet connection..."

echo "Pulling embedding model (nomic-embed-text)..."
ollama pull nomic-embed-text

echo "Pulling chat model (llama3.2)..."
ollama pull llama3.2

echo ""
echo "Setup complete!"
echo "You can now run: python document_store.py"
