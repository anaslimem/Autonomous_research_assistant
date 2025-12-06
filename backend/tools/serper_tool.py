import httpx
from typing import Optional
from google.adk.tools import FunctionTool
import os


SERPER_API_KEY = os.getenv("SERPER_API_KEY")
SERPER_BASE_URL = "https://google.serper.dev"


def search_google(
    query: str,
    num_results: int = 10,
    search_type: str = "search"
) -> dict:
    """
    Search Google using Serper API.
    
    Args:
        query: Search query string
        num_results: Number of results to return (default: 10, max: 100)
        search_type: Type of search - "search", "news", "images", "places"
    
    Returns:
        Dictionary containing search results
    """
    num_results = min(num_results, 100)
    
    try:
        headers = {
            "X-API-KEY": SERPER_API_KEY,
            "Content-Type": "application/json"
        }
        
        payload = {
            "q": query,
            "num": num_results
        }
        
        endpoint = f"{SERPER_BASE_URL}/{search_type}"
        
        with httpx.Client(timeout=30.0) as client:
            response = client.post(endpoint, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
        
        # Parse organic results
        results = []
        for item in data.get("organic", []):
            results.append({
                "title": item.get("title"),
                "url": item.get("link"),
                "snippet": item.get("snippet"),
                "position": item.get("position"),
            })
        
        return {
            "status": "success",
            "query": query,
            "search_type": search_type,
            "total_results": len(results),
            "results": results,
            "knowledge_graph": data.get("knowledgeGraph"),
            "answer_box": data.get("answerBox"),
        }
        
    except httpx.HTTPStatusError as e:
        return {
            "status": "error",
            "query": query,
            "error": f"HTTP {e.response.status_code}: {e.response.text}",
            "results": []
        }
    except Exception as e:
        return {
            "status": "error",
            "query": query,
            "error": str(e),
            "results": []
        }


def search_google_news(
    query: str,
    num_results: int = 10
) -> dict:
    """
    Search Google News using Serper API.
    
    Args:
        query: Search query string
        num_results: Number of results to return (default: 10)
    
    Returns:
        Dictionary containing news results
    """
    num_results = min(num_results, 100)
    
    try:
        headers = {
            "X-API-KEY": SERPER_API_KEY,
            "Content-Type": "application/json"
        }
        
        payload = {
            "q": query,
            "num": num_results
        }
        
        with httpx.Client(timeout=30.0) as client:
            response = client.post(f"{SERPER_BASE_URL}/news", json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
        
        results = []
        for item in data.get("news", []):
            results.append({
                "title": item.get("title"),
                "url": item.get("link"),
                "snippet": item.get("snippet"),
                "source": item.get("source"),
                "date": item.get("date"),
                "image_url": item.get("imageUrl"),
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


def search_google_scholar(
    query: str,
    num_results: int = 10
) -> dict:
    """
    Search Google Scholar using Serper API.
    
    Args:
        query: Academic search query string
        num_results: Number of results to return (default: 10)
    
    Returns:
        Dictionary containing scholarly article results
    """
    num_results = min(num_results, 100)
    
    try:
        headers = {
            "X-API-KEY": SERPER_API_KEY,
            "Content-Type": "application/json"
        }
        
        payload = {
            "q": query,
            "num": num_results
        }
        
        with httpx.Client(timeout=30.0) as client:
            response = client.post(f"{SERPER_BASE_URL}/scholar", json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
        
        results = []
        for item in data.get("organic", []):
            results.append({
                "title": item.get("title"),
                "url": item.get("link"),
                "snippet": item.get("snippet"),
                "publication_info": item.get("publicationInfo"),
                "cited_by": item.get("citedBy"),
                "year": item.get("year"),
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


# Create ADK FunctionTools
search_google_tool = FunctionTool(search_google)
search_google_news_tool = FunctionTool(search_google_news)
search_google_scholar_tool = FunctionTool(search_google_scholar)

# Export all tools
serper_tools = [search_google_tool, search_google_news_tool, search_google_scholar_tool]
