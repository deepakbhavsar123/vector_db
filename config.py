# Configuration for Document Store
import os

# Directories
FILES_DIRECTORY = "Files"
DATABASE_PATH = "chroma_db"

# Ollama settings
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2"

# Text processing settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Search settings
DEFAULT_SEARCH_RESULTS = 5

# Supported file extensions
SUPPORTED_EXTENSIONS = {'.docx', '.pdf', '.txt', '.csv'}

# Logging level
LOG_LEVEL = "INFO"
