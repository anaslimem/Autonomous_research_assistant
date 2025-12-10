from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from typing import Union
import os, logging

# Set up logging
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Get environment variables
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL")
EMBEDDING_DIMENSION = os.getenv("EMBEDDING_DIMENSION")

# Validate required environment variables
required_vars = {
    "EMBEDDING_MODEL": EMBEDDING_MODEL_NAME,
    "EMBEDDING_DIMENSION": EMBEDDING_DIMENSION
}
missing = [name for name, value in required_vars.items() if not value]
if missing:
    raise EnvironmentError(f"Missing required environment variables: {missing}")

# Convert types after validation
EMBEDDING_DIMENSION = int(EMBEDDING_DIMENSION)

# Initialize embedding model
try:
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, trust_remote_code=True)
    logger.info(f"Loaded embedding model: {EMBEDDING_MODEL_NAME}")
except Exception as e:
    logger.error(f"Failed to load embedding model '{EMBEDDING_MODEL_NAME}': {e}")
    raise e


def embed_documents(texts: Union[str, list[str]]) -> list:
    """
    Generate embeddings for documents (for storing in Qdrant).
    Uses 'search_document:' prefix for optimal retrieval with Nomic model.
    
    Args:
        texts: Single text or list of texts to embed
    Returns:
        List of embedding vectors (list of lists if multiple texts)
    """
    if isinstance(texts, str):
        texts = [texts]
    
    # Add Nomic document prefix
    prefixed_texts = [f"search_document: {text}" for text in texts]
    
    embeddings = embedder.encode(prefixed_texts)
    
    # Return single embedding if single input, else list
    if len(embeddings) == 1:
        return embeddings[0].tolist()
    return [emb.tolist() for emb in embeddings]


def embed_query(query: str) -> list:
    """
    Generate embedding for a search query (for searching Qdrant).
    Uses 'search_query:' prefix for optimal retrieval with Nomic model.
    
    Args:
        query: Search query text
    Returns:
        Embedding vector as list
    """
    # Add Nomic query prefix
    prefixed_query = f"search_query: {query}"
    
    embedding = embedder.encode(prefixed_query)
    return embedding.tolist()

