from backend.tools.scrapper_tool import scrape_url
from backend.ingestion.chunker import chunk_and_embed
from backend.ingestion.entity_extractor import extract_entities
from backend.storage.qdrant_store import store_chunks, init_collection
from backend.storage.neo4j_store import store_article_with_entities
import logging

# Set up logging
logger = logging.getLogger(__name__)

def ingest_url(url: str) -> dict:
    """
    Ingest a document from a URL: scrape, chunk, embed, and store in Qdrant.
    
    Args:
        url: The URL of the document to ingest.
    Returns:
        dict: Status and details of the ingestion process.
    """
    # Scrape
    scraped = scrape_url(url)

    # Check if scraping was successful
    if scraped["status"] != "success":
        logger.error(f"Scraping failed for URL {url}: {scraped['error']}")
        return {
            "status": "error",
            "error": f"Scraping failed: {scraped['error']}",
            "result": None
        }
    
    result = scraped["result"]

    # Chunk and embed
    chunks = chunk_and_embed(result["text"])

    if not chunks:
        logger.error(f"No chunks created for URL {url}")
        return {
            "status": "error",
            "error": "No chunks created from the document",
            "result": None
        }
    
    # Store in Qdrant
    metadata = {
        "source_url": result["url"],
        "title": result["title"],
        "domain": result["domain"],
        "scraped_at": result["scraped_at"]
    }

    init_collection()  # Ensure collection exists
    count = store_chunks(chunks, metadata)
    logger.info(f"Stored {count} chunks for URL {url}")

    # Extract entities using Gemini
    entities = extract_entities(result["text"], title=result["title"])
    
    # Store in Neo4j
    article_data = {
        "title": result["title"],
        "source_url": result["url"],
        "domain": result["domain"],
        "authors": entities.get("authors", []),
        "topics": entities.get("topics", []),
        "technologies": entities.get("technologies", []),
        "companies": entities.get("companies", []),
        "concepts": entities.get("concepts", [])
    }
    
    try:
        store_article_with_entities(article_data)
        logger.info(f"Stored entities in Neo4j for URL {url}")
    except Exception as e:
        logger.error(f"Failed to store entities in Neo4j: {e}")

    return {
        "status": "success",
        "url": url,
        "title": result["title"],
        "chunk_count": count,
        "entities": entities
    }

