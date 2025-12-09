from backend.storage.qdrant_store import search_similar
from backend.ingestion.embedder import embed_query
from backend.storage.neo4j_store import neo4j_driver
from typing import Optional
import logging

# Set up logging
logger = logging.getLogger(__name__)

def hybrid_search(query: str, limit: int = 5) -> str:
    """
    Perform a hybrid search combining vector semantic search (Qdrant) with 
    knowledge graph exploration (Neo4j) for comprehensive results.
    
    This function:
    1. Searches the vector database for semantically similar documents
    2. Extracts key entities/topics from the query
    3. Finds related articles and entities in the knowledge graph
    4. Combines and deduplicates results
    
    Args:
        query: The search query text
        limit: Number of results per source (default: 5)
    Returns:
        str: Formatted string with combined results from both sources
    """
    logger.info(f"Performing hybrid search for: '{query}'")
    
    output = "HYBRID SEARCH RESULTS\n\n"
    has_results = False
    
    # Part 1: Vector Search (Semantic Similarity) 
    output += "VECTOR DATABASE RESULTS (Semantic Search)\n"
    output += "-" * 50 + "\n"
    
    try:
        query_vector = embed_query(query)
        vector_results = search_similar(query_vector, limit=limit)
        
        if vector_results:
            has_results = True
            for i, r in enumerate(vector_results, 1):
                output += f"[{i}] Source: {r.get('title', 'Unknown')}\n"
                output += f"    Similarity Score: {r['score']:.3f}\n"
                output += f"    Content: {r['text'][:400]}...\n\n"
        else:
            output += "No semantically similar documents found.\n\n"
    except Exception as e:
        logger.error(f"Vector search error: {e}")
        output += f"Vector search encountered an error: {e}\n\n"
    
    # Part 2: Knowledge Graph Search (Entity Relations) 
    output += "\nKNOWLEDGE GRAPH RESULTS (Entity Relationships)\n"
    output += "-" * 50 + "\n"
    
    # Use key words from the query to search the graph
    query_terms = [term.strip() for term in query.split() if len(term.strip()) > 3]
    
    try:
        with neo4j_driver.session() as session:
            # Search for articles mentioning query terms
            articles_query = """
            MATCH (a:Article)-[r]->(e)
            WHERE any(term IN $terms WHERE toLower(e.name) CONTAINS toLower(term))
               OR any(term IN $terms WHERE toLower(a.name) CONTAINS toLower(term))
            WITH a, collect(DISTINCT {type: labels(e)[0], name: e.name, rel: type(r)}) as entities
            RETURN DISTINCT a.name as title, a.url as url, entities
            LIMIT $limit
            """
            result = session.run(articles_query, terms=query_terms, limit=limit)
            records = list(result)
            
            if records:
                has_results = True
                output += "\n**Related Articles from Knowledge Graph:**\n"
                for record in records:
                    output += f"\n• {record['title']}\n"
                    if record['url']:
                        output += f"  URL: {record['url']}\n"
                    if record['entities']:
                        entities_str = ", ".join([f"{e['name']} ({e['type']})" for e in record['entities'][:5]])
                        output += f"  Connected Entities: {entities_str}\n"
            else:
                output += "No related articles found in knowledge graph.\n"
            
            # Also find related concepts/entities
            concepts_query = """
            MATCH (e)-[r]-(related)
            WHERE any(term IN $terms WHERE toLower(e.name) CONTAINS toLower(term))
            RETURN DISTINCT labels(e)[0] as entity_type, e.name as entity_name,
                   type(r) as relationship, labels(related)[0] as related_type, 
                   related.name as related_name
            LIMIT 15
            """
            concepts_result = session.run(concepts_query, terms=query_terms)
            concept_records = list(concepts_result)
            
            if concept_records:
                has_results = True
                output += "\n**Entity Relationships:**\n"
                for record in concept_records:
                    output += f"  • ({record['entity_type']}) {record['entity_name']} "
                    output += f"-[{record['relationship']}]-> "
                    output += f"({record['related_type']}) {record['related_name']}\n"
                    
    except Exception as e:
        logger.error(f"Knowledge graph search error: {e}")
        output += f"Knowledge graph search encountered an error: {e}\n"
    
    # Summary
    output += "\n" + "=" * 50 + "\n"
    if has_results:
        output += "Hybrid search completed successfully. Results from both vector DB and knowledge graph are shown above.\n"
    else:
        output += "No results found in either vector database or knowledge graph.\n"
    
    return output
