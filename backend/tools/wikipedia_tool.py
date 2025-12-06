import httpx
from typing import Optional
from google.adk.tools import FunctionTool


BASE_URL = "https://en.wikipedia.org/api/rest_v1"
SEARCH_URL = "https://en.wikipedia.org/w/api.php"

# Wikipedia requires a User-Agent header to identify the client
HEADERS = {
    "User-Agent": "ResearchAssistantBot/1.0 (Educational project; Python/httpx)"
}


def search_wikipedia(
    query: str,
    max_results: int = 10
) -> dict:
    """
    Search Wikipedia for articles matching the query.
    
    Args:
        query: Search query string (e.g., "quantum computing", "machine learning")
        max_results: Maximum number of results to return (default: 10, max: 50)
    
    Returns:
        Dictionary containing search results with article titles and snippets
    """
    max_results = min(max_results, 10)
    
    try:
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": max_results,
            "format": "json",
            "utf8": 1
        }
        
        with httpx.Client(timeout=30.0, headers=HEADERS) as client:
            response = client.get(SEARCH_URL, params=params)
            response.raise_for_status()
            data = response.json()
        
        results = []
        for item in data.get("query", {}).get("search", []):
            results.append({
                "title": item.get("title"),
                "snippet": item.get("snippet", "").replace("<span class=\"searchmatch\">", "").replace("</span>", ""),
                "word_count": item.get("wordcount"),
                "page_id": item.get("pageid"),
                "url": f"https://en.wikipedia.org/wiki/{item.get('title', '').replace(' ', '_')}"
            })
        
        return {
            "status": "success",
            "query": query,
            "total_results": len(results),
            "results": results
        }
        
    except Exception as e:
        return {
            "status": "error",
            "query": query,
            "error": str(e),
            "results": []
        }


def get_wikipedia_summary(title: str) -> dict:
    """
    Get the summary of a Wikipedia article.
    
    Args:
        title: The title of the Wikipedia article (e.g., "Quantum computing")
    
    Returns:
        Dictionary containing the article summary and metadata
    """
    try:
        # URL encode the title
        encoded_title = title.replace(" ", "_")
        url = f"{BASE_URL}/page/summary/{encoded_title}"
        
        with httpx.Client(timeout=30.0, headers=HEADERS) as client:
            response = client.get(url)
            response.raise_for_status()
            data = response.json()
        
        return {
            "status": "success",
            "title": data.get("title"),
            "result": {
                "title": data.get("title"),
                "description": data.get("description"),
                "extract": data.get("extract"),
                "url": data.get("content_urls", {}).get("desktop", {}).get("page"),
                "thumbnail": data.get("thumbnail", {}).get("source"),
                "type": data.get("type"),
                "page_id": data.get("pageid"),
            }
        }
        
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return {
                "status": "error",
                "title": title,
                "error": "Article not found",
                "result": None
            }
        return {
            "status": "error",
            "title": title,
            "error": str(e),
            "result": None
        }
    except Exception as e:
        return {
            "status": "error",
            "title": title,
            "error": str(e),
            "result": None
        }


def get_wikipedia_content(title: str) -> dict:
    """
    Get the full content of a Wikipedia article in plain text.
    
    Args:
        title: The title of the Wikipedia article (e.g., "Quantum computing")
    
    Returns:
        Dictionary containing the full article content
    """
    try:
        params = {
            "action": "query",
            "titles": title,
            "prop": "extracts",
            "explaintext": True,
            "format": "json",
            "utf8": 1
        }
        
        with httpx.Client(timeout=30.0, headers=HEADERS) as client:
            response = client.get(SEARCH_URL, params=params)
            response.raise_for_status()
            data = response.json()
        
        pages = data.get("query", {}).get("pages", {})
        
        for page_id, page_data in pages.items():
            if page_id == "-1":
                return {
                    "status": "error",
                    "title": title,
                    "error": "Article not found",
                    "result": None
                }
            
            return {
                "status": "success",
                "title": page_data.get("title"),
                "result": {
                    "title": page_data.get("title"),
                    "page_id": page_data.get("pageid"),
                    "content": page_data.get("extract", ""),
                    "url": f"https://en.wikipedia.org/wiki/{page_data.get('title', '').replace(' ', '_')}"
                }
            }
        
        return {
            "status": "error",
            "title": title,
            "error": "No content found",
            "result": None
        }
        
    except Exception as e:
        return {
            "status": "error",
            "title": title,
            "error": str(e),
            "result": None
        }


# Create ADK FunctionTools
search_wikipedia_tool = FunctionTool(search_wikipedia)
get_wikipedia_summary_tool = FunctionTool(get_wikipedia_summary)
get_wikipedia_content_tool = FunctionTool(get_wikipedia_content)

# Export all tools
wikipedia_tools = [search_wikipedia_tool, get_wikipedia_summary_tool, get_wikipedia_content_tool]
