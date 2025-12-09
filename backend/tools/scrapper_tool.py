import trafilatura
from urllib.parse import urlparse
from datetime import datetime
import logging

# Set up logging
logger = logging.getLogger(__name__)

def scrape_url(url: str) -> dict:
    """
    Scrape content from a URL using Trafilatura.
    
    Args:
        url: The URL to scrape.
    Returns:
        dict: A dictionary with scraped content and metadata.
    """
    # Fetch the HTML
    downloaded = trafilatura.fetch_url(url)

    if not downloaded:
        logger.error(f"Failed to download content from URL: {url}")
        return {"status": "error", "error": "Failed to download content", "result": None}
    
    # Extract content
    text = trafilatura.extract(downloaded)
    metadata = trafilatura.extract_metadata(downloaded)

    if not text:
        logger.error(f"No extractable text found at URL: {url}")
        return {"status": "error", "error": "No extractable text found", "result": None}
    
    # Build result
    domain = urlparse(url).netloc

    result = {
        "url": url,
        "title": metadata.title if metadata else None,
        "text": text,
        "domain": domain,
        "scraped_at": datetime.utcnow().isoformat()
    }

    return {"status": "success", "error": None, "result": result}