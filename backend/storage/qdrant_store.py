from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct
from dotenv import load_dotenv
import os
import logging
import uuid

# Set up logging
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Get environment variables 
QDRANT_HOST = os.getenv("QDRANT_URL")
QDRANT_PORT = os.getenv("QDRANT_PORT")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")
EMBEDDING_DIMENSION = os.getenv("EMBEDDING_DIMENSION")

# Validate required environment variables
required_vars = {
    "QDRANT_URL": QDRANT_HOST,
    "QDRANT_PORT": QDRANT_PORT,
    "QDRANT_COLLECTION": QDRANT_COLLECTION,
    "EMBEDDING_DIMENSION": EMBEDDING_DIMENSION
}

missing = [name for name, value in required_vars.items() if not value]
if missing:
    raise EnvironmentError(f"Missing required environment variables: {missing}")

# Convert types after validation
QDRANT_PORT = int(QDRANT_PORT)
EMBEDDING_DIMENSION = int(EMBEDDING_DIMENSION)

# Initialize Qdrant client Connection
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
logger.info(f"Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")

# Ensure the collection exists
def init_collection():
    """Initialize collection if it doesn't exist."""
    try:
        qdrant_client.get_collection(collection_name=QDRANT_COLLECTION)
        logger.info(f"Collection '{QDRANT_COLLECTION}' already exists")
    except Exception:
        qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=models.VectorParams(
                size=EMBEDDING_DIMENSION,
                distance=models.Distance.COSINE
            )
        )
        logger.info(f"Created new collection '{QDRANT_COLLECTION}' with dimension {EMBEDDING_DIMENSION}")

def store_chunks(chunks: list[dict], metadata: dict) -> int:
    """
    Store document chunks in Qdrant.
    
    Args:
        chunks: List of dicts with 'text', 'embedding', and 'index'
        metadata: Additional metadata to store with each chunk
    Returns:
        Number of chunks stored
    """
    if not chunks:
        logger.warning("No chunks to store")
        return 0
    
    points = []
    for chunk in chunks:
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=chunk['embedding'],
            payload={
                "text": chunk['text'],
                "index": chunk['index'],
                **metadata
            }
        )
        points.append(point)
    qdrant_client.upsert(
        collection_name=QDRANT_COLLECTION,
        points=points
    )
    logger.info(f"Stored {len(points)} chunks in collection '{QDRANT_COLLECTION}'")
    return len(points)

def search_similar(query_embedding: list, limit: int = 5, score_threshold: float = None) -> list[dict]:
    """
    Search for similar document chunks in Qdrant.
    
    Args:
        query_embedding: Embedding vector of the search query
        limit: Number of results to return (default: 5)
        score_threshold: Minimum similarity score to include result (optional)
    Returns:
        List of dicts with 'text', 'index', and 'score'
    """
    search_result = qdrant_client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_embedding,
        limit=limit,
        score_threshold=score_threshold
    )
    
    results = []
    for hit in search_result.points:
        results.append({
            "text": hit.payload.get("text"),
            "index": hit.payload.get("index"),
            "score": hit.score,
            "source_url": hit.payload.get("source_url"),
            "title": hit.payload.get("title"),
            "domain": hit.payload.get("domain"),
            "scraped_at": hit.payload.get("scraped_at")
        })
    
    logger.info(f"Found {len(results)} similar chunks in collection '{QDRANT_COLLECTION}'")
    return results