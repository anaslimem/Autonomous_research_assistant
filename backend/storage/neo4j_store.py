from neo4j import GraphDatabase
from dotenv import load_dotenv
import os, logging

# Set up logging
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Get environment variables
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Validate required environment variables
required_vars = {
    "NEO4J_URI": NEO4J_URI,
    "NEO4J_USER": NEO4J_USER,
    "NEO4J_PASSWORD": NEO4J_PASSWORD
}

missing = [name for name, value in required_vars.items() if not value]
if missing:
    raise EnvironmentError(f"Missing required environment variables: {missing}")

# Initialize Neo4j driver
neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
logger.info(f"Connected to Neo4j at {NEO4J_URI}")

# Define entity types for articles/blogs
ENTITY_TYPES = ["Article", "Author", "Topic", "Technology", "Company", "Concept"]

# Define relationship types
RELATIONSHIP_TYPES = {
    "WRITTEN_BY": ("Article", "Author"),
    "ABOUT_TOPIC": ("Article", "Topic"),
    "MENTIONS": ("Article", ["Technology", "Company", "Concept"]),
    "RELATED_TO": ("Topic", "Topic"),
    "USES": ("Technology", "Technology"),
    "DEVELOPED_BY": ("Technology", "Company")
}


def verify_connection():
    """Verify Neo4j connection is working."""
    try:
        with neo4j_driver.session() as session:
            result = session.run("RETURN 1 AS connected")
            record = result.single()
            if record and record["connected"] == 1:
                logger.info("Neo4j connection verified successfully")
                return True
    except Exception as e:
        logger.error(f"Neo4j connection failed: {e}")
        return False


def create_entity(entity_type: str, name: str, properties: dict = None):
    """
    Create a node/entity in Neo4j.
    
    Args:
        entity_type: One of ENTITY_TYPES 
        name: The name/title of the entity
        properties: Additional properties for the node
    """
    if entity_type not in ENTITY_TYPES:
        raise ValueError(f"Invalid entity type: {entity_type}. Must be one of {ENTITY_TYPES}")
    
    props = properties or {}
    props["name"] = name
    
    with neo4j_driver.session() as session:
        # MERGE prevents duplicates - creates only if doesn't exist
        query = f"""
        MERGE (n:{entity_type} {{name: $name}})
        SET n += $props
        RETURN n
        """
        result = session.run(query, name=name, props=props)
        logger.debug(f"Created/Updated {entity_type}: {name}")
        return result.single()


def create_relationship(from_type: str, from_name: str, relationship: str, to_type: str, to_name: str):
    """
    Create a relationship between two entities.
    
    Args:
        from_type: Entity type of source node
        from_name: Name of source node
        relationship: Relationship type 
        to_type: Entity type of target node
        to_name: Name of target node
    """
    with neo4j_driver.session() as session:
        query = f"""
        MATCH (a:{from_type} {{name: $from_name}})
        MATCH (b:{to_type} {{name: $to_name}})
        MERGE (a)-[r:{relationship}]->(b)
        RETURN a, r, b
        """
        result = session.run(query, from_name=from_name, to_name=to_name)
        logger.debug(f"Created relationship: ({from_name})-[{relationship}]->({to_name})")
        return result.single()


def store_article_with_entities(article_data: dict):
    """
    Store an article and all its extracted entities with relationships.
    
    Args:
        article_data: Dict containing:
            - title: Article title
            - url: Source URL
            - author: Author name (optional)
            - topics: List of topics
            - technologies: List of technologies mentioned
            - companies: List of companies mentioned
            - concepts: List of concepts mentioned
    """
    title = article_data.get("title")
    url = article_data.get("source_url", "")
    
    # Create Article node
    create_entity("Article", title, {"url": url, "domain": article_data.get("domain", "")})
    
    # Create Authors and relationships
    for author in article_data.get("authors", []):
        create_entity("Author", author)
        create_relationship("Article", title, "WRITTEN_BY", "Author", author)
    
    # Create Topics and relationships
    for topic in article_data.get("topics", []):
        create_entity("Topic", topic)
        create_relationship("Article", title, "ABOUT_TOPIC", "Topic", topic)
    
    # Create Technologies and relationships
    for tech in article_data.get("technologies", []):
        create_entity("Technology", tech)
        create_relationship("Article", title, "MENTIONS", "Technology", tech)
    
    # Create Companies and relationships
    for company in article_data.get("companies", []):
        create_entity("Company", company)
        create_relationship("Article", title, "MENTIONS", "Company", company)
    
    # Create Concepts and relationships
    for concept in article_data.get("concepts", []):
        create_entity("Concept", concept)
        create_relationship("Article", title, "MENTIONS", "Concept", concept)
    
    logger.info(f"Stored article '{title}' with all entities")


def close_connection():
    """Close the Neo4j driver connection."""
    neo4j_driver.close()
    logger.info("Neo4j connection closed")


