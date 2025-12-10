from chonkie import SemanticChunker
from chonkie.embeddings import SentenceTransformerEmbeddings
from backend.ingestion.embedder import embed_documents, embedder
from dotenv import load_dotenv
import logging
import os

# Set up logging
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD"))

# Wrap our already-loaded embedder for Chonkie
# This avoids Chonkie trying to load the model again without trust_remote_code
chonkie_embeddings = SentenceTransformerEmbeddings(model=embedder)

# Initialize SemanticChunker with our pre-loaded embedding model
semantic_chunker = SemanticChunker(
    embedding_model=chonkie_embeddings,
    chunk_size=CHUNK_SIZE,
    similarity_threshold=SIMILARITY_THRESHOLD
)
logger.info(f"Initialized SemanticChunker (chunk_size={CHUNK_SIZE}, similarity={SIMILARITY_THRESHOLD})")


def chunk_text(text: str) -> list[dict]:
    """
    Chunk a document into semantic chunks.
    
    Args:
        text: The full text of the document to be chunked.
    Returns:
        List of dicts with 'text' and 'index' for each chunk.
    """
    if not text or not text.strip():
        logger.warning("Empty text provided for chunking")
        return []
    
    # Chunk the document
    chunks = semantic_chunker.chunk(text)
    logger.info(f"Document chunked into {len(chunks)} semantic chunks")
    
    # Extract chunk data
    chunk_list = []
    for idx, chunk in enumerate(chunks):
        chunk_list.append({
            "text": chunk.text,
            "index": idx,
            "token_count": chunk.token_count if hasattr(chunk, 'token_count') else None
        })
    
    return chunk_list


def chunk_and_embed(text: str) -> list[dict]:
    """
    Chunk a document and generate embeddings for each chunk.
    
    Args:
        text: The full text of the document to be chunked and embedded.
    Returns:
        List of dicts with 'text', 'index', and 'embedding' for each chunk.
    """
    # First chunk the text
    chunks = chunk_text(text)
    
    if not chunks:
        return []
    
    # Extract texts for batch embedding
    texts = [chunk["text"] for chunk in chunks]
    
    # Generate embeddings in batch (more efficient)
    embeddings = embed_documents(texts)
    
    # Handle single chunk case (embed_documents returns single list, not list of lists)
    if len(chunks) == 1:
        embeddings = [embeddings]
    
    # Combine chunks with embeddings
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding
    
    logger.info(f"Generated embeddings for {len(chunks)} chunks")
    
    return chunks
