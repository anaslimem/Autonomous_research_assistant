import arxiv
from typing import Optional
from google.adk.tools import FunctionTool


def search_arxiv(
    query: str,
    max_results: int = 10,
    sort_by: str = "relevance"
) -> dict:
    """
    Search arXiv for academic papers.
    
    Args:
        query: Search query string (e.g., "quantum computing", "machine learning")
        max_results: Maximum number of results to return (default: 10, max: 50)
        sort_by: Sort order - "relevance", "lastUpdatedDate", or "submittedDate"
    
    Returns:
        Dictionary containing search results with paper metadata
    """
    # Validate inputs
    max_results = min(max_results, 10)  # Cap at 10
    
    # Map sort options
    sort_options = {
        "relevance": arxiv.SortCriterion.Relevance,
        "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
        "submittedDate": arxiv.SortCriterion.SubmittedDate,
    }
    sort_criterion = sort_options.get(sort_by, arxiv.SortCriterion.Relevance)
    
    try:
        # Create search client
        client = arxiv.Client()
        
        # Build search query
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=sort_criterion
        )
        
        # Execute search and collect results
        results = []
        for paper in client.results(search):
            results.append({
                "title": paper.title,
                "authors": [author.name for author in paper.authors],
                "abstract": paper.summary,
                "published": paper.published.strftime("%Y-%m-%d"),
                "updated": paper.updated.strftime("%Y-%m-%d"),
                "arxiv_id": paper.entry_id.split("/")[-1],
                "url": paper.entry_id,
                "pdf_url": paper.pdf_url,
                "primary_category": paper.primary_category,
                "categories": paper.categories,
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


def get_arxiv_paper(arxiv_id: str) -> dict:
    """
    Get detailed information about a specific arXiv paper by its ID.
    
    Args:
        arxiv_id: The arXiv paper ID (e.g., "2301.07041" or "2301.07041v1")
    
    Returns:
        Dictionary containing detailed paper information
    """
    try:
        client = arxiv.Client()
        
        # Search by ID
        search = arxiv.Search(id_list=[arxiv_id])
        
        # Get the paper
        paper = next(client.results(search), None)
        
        if paper is None:
            return {
                "status": "error",
                "arxiv_id": arxiv_id,
                "error": "Paper not found",
                "result": None
            }
        
        return {
            "status": "success",
            "arxiv_id": arxiv_id,
            "result": {
                "title": paper.title,
                "authors": [author.name for author in paper.authors],
                "abstract": paper.summary,
                "published": paper.published.strftime("%Y-%m-%d"),
                "updated": paper.updated.strftime("%Y-%m-%d"),
                "url": paper.entry_id,
                "pdf_url": paper.pdf_url,
                "primary_category": paper.primary_category,
                "categories": paper.categories,
                "comment": paper.comment,
                "journal_ref": paper.journal_ref,
                "doi": paper.doi,
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "arxiv_id": arxiv_id,
            "error": str(e),
            "result": None
        }


# Create ADK FunctionTools
search_arxiv_tool = FunctionTool(search_arxiv)
get_arxiv_paper_tool = FunctionTool(get_arxiv_paper)

# Export all tools
arxiv_tools = [search_arxiv_tool, get_arxiv_paper_tool]
