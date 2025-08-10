# Document Store with Ollama Integration

A Python application that stores documents from a directory and enables intelligent querying using Ollama (free open-source LLM).

## Features

- **Document Processing**: Supports DOCX, PDF, TXT, and CSV files
- **Vector Storage**: Uses ChromaDB for efficient similarity search
- **LLM Integration**: Powered by Ollama for intelligent responses
- **Chunking**: Automatically splits large documents into manageable chunks
- **CLI Interface**: Easy-to-use command-line interface
- **Interactive Mode**: Real-time question-answering

## Prerequisites

1. **Python 3.8+**
2. **Ollama**: Install from [https://ollama.ai/](https://ollama.ai/)

### Installing Ollama (Linux)
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

## Installation

1. **Clone or download the files**
2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the setup script**:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

This will:
- Install Python dependencies
- Start Ollama service
- Download required models (`nomic-embed-text` for embeddings, `llama3.2` for chat)

## Usage

### Method 1: Using the CLI

#### Store all files from the Files directory:
```bash
python cli.py store --all
```

#### Store a specific file:
```bash
python cli.py store --file "Files/document.pdf"
```

#### Query documents:
```bash
python cli.py query "What is mentioned about experience?"
```

#### Search for specific content:
```bash
python cli.py search "skills"
```

#### Show database statistics:
```bash
python cli.py stats
```

#### Start interactive mode:
```bash
python cli.py interactive
```

### Method 2: Using the main script

```bash
python document_store.py
```

This will:
1. Process all files in the `Files` directory
2. Store them in the vector database
3. Start an interactive query session

### Method 3: Programmatic usage

```python
from document_store import DocumentStore

# Initialize
doc_store = DocumentStore()

# Store files
doc_store.store_all_files()

# Query
response = doc_store.query_with_llm("What skills are mentioned?")
print(response)

# Search
results = doc_store.search_documents("experience", n_results=3)
for result in results:
    print(result['document'])
```

## Configuration

Edit `config.py` to customize:
- File directories
- Ollama models
- Chunk sizes
- Search parameters

## Supported File Types

- **.docx**: Microsoft Word documents
- **.pdf**: PDF files
- **.txt**: Plain text files
- **.csv**: CSV files

## How It Works

1. **Document Processing**: Extracts text from various file formats
2. **Text Chunking**: Splits large documents into overlapping chunks for better retrieval
3. **Embedding Generation**: Uses Ollama's `nomic-embed-text` model to generate vector embeddings
4. **Vector Storage**: Stores embeddings in ChromaDB for fast similarity search
5. **Retrieval**: Finds relevant document chunks based on query similarity
6. **Response Generation**: Uses Ollama's `llama3.2` model to generate contextual responses

## Troubleshooting

### Ollama not available
- Make sure Ollama is installed and running: `ollama serve`
- Check if models are downloaded: `ollama list`
- Pull required models: `ollama pull nomic-embed-text && ollama pull llama3.2`

### File processing errors
- Ensure files are not corrupted
- Check file permissions
- Verify supported file formats

### Memory issues
- Reduce chunk size in `config.py`
- Process files one at a time instead of batch processing

## Example Queries

Based on the resume file in your Files directory, you can ask:

- "What is the candidate's experience?"
- "What programming languages does the candidate know?"
- "Tell me about the education background"
- "What projects has the candidate worked on?"
- "What are the key skills mentioned?"

## Directory Structure

```
vector_db/
├── Files/                    # Directory containing documents to process
│   └── Deepak_ms_Zensar_resume.docx
├── chroma_db/               # ChromaDB storage (created automatically)
├── document_store.py        # Main application
├── cli.py                   # Command-line interface
├── config.py               # Configuration settings
├── requirements.txt        # Python dependencies
├── setup.sh               # Installation script
└── README.md              # This file
```

## Advanced Usage

### Custom Models
You can use different Ollama models by modifying the configuration:

```python
# For embeddings
doc_store.ollama_client.generate_embeddings(text, model="custom-embed-model")

# For chat
doc_store.query_with_llm(question, model="custom-chat-model")
```

### Batch Processing
For large document collections:

```python
import os
from pathlib import Path

doc_store = DocumentStore()
for file_path in Path("large_files").glob("**/*"):
    if file_path.is_file():
        doc_store.store_file(str(file_path))
```

## Performance Tips

1. **Use appropriate chunk sizes**: Smaller chunks for precise search, larger for context
2. **Regular updates**: Re-process modified files to keep embeddings current
3. **Model selection**: Choose models based on your hardware capabilities
4. **Batch operations**: Process multiple files in batches for efficiency

## License

This project is open source. Feel free to modify and distribute.
