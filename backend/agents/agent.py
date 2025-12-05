from google.adk.agents import LlmAgent
from dotenv import load_dotenv
import os
import logging
import sys
from datetime import datetime
from pathlib import Path

# Load environment variables from a .env file
load_dotenv()

# Logging Configuration

# Create logs directory if it doesn't exist
log_dir = Path(__file__).parent.parent.parent / "logs"
log_dir.mkdir(exist_ok=True)

# Configure logging format
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Set log level from environment (default to INFO)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()

# Create root logger
logging.basicConfig(
    level=getattr(logging, log_level),
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,
    handlers=[
        # Console handler with colored output
        logging.StreamHandler(sys.stdout),
        # File handler for persistent logs
        logging.FileHandler(log_dir / "agents.log", mode="a", encoding="utf-8"),
    ]
)

# Create agent-specific loggers
orchestrator_logger = logging.getLogger("agent.orchestrator")
planning_logger = logging.getLogger("agent.planning")
summarization_logger = logging.getLogger("agent.summarization")
retrieval_logger = logging.getLogger("agent.retrieval")
tool_use_logger = logging.getLogger("agent.tool_use")

# Main logger for this module
logger = logging.getLogger("agent.main")


def validate_config():
    """Validate required configuration is present."""
    required_vars = ["GOOGLE_API_KEY"]
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        logger.error(f"Missing required environment variables: {missing}")
        raise EnvironmentError(f"Missing required environment variables: {missing}")
    
    logger.info("✓ Configuration validated successfully")
    logger.info(f"  - Google API Key: {'*' * 10}{os.getenv('GOOGLE_API_KEY')[-4:]}")
    logger.info(f"  - Log Level: {log_level}")

# Validate on module load
validate_config()

logger.info("Initializing Research Assistant Agents...")

# Initialize the Planning agent
planning_logger.info("Initializing Planning Agent...") 
planning_agent = LlmAgent(
    model=os.getenv("MODEL"),
    name="planning_agent",
    description="Breaks down complex research queries into structured, actionable execution plans.",
    instruction="""You are a Task Decomposition Specialist that creates research execution plans.

When a user submits a research query:
1. Analyze the query to understand scope, complexity, and objectives.
2. Decompose the query into specific, executable subtasks.
3. Assign each subtask to the appropriate agent (Retrieval, Summarization, ToolUse).
4. Define task dependencies and execution order (sequential or parallel).
5. Estimate time and prioritize tasks by importance.
6. Return a structured execution plan.

Example Query: "What are the latest advances in quantum computing for drug discovery?"
Example Response:
{
    "objective": "Research quantum computing applications in drug discovery",
    "tasks": [
        {"id": 1, "agent": "ToolUse", "task": "Search arXiv for quantum computing drug discovery papers", "priority": "high"},
        {"id": 2, "agent": "ToolUse", "task": "Search PubMed for computational drug discovery methods", "priority": "high"},
        {"id": 3, "agent": "Retrieval", "task": "Query knowledge base for quantum algorithms", "depends_on": []},
        {"id": 4, "agent": "Summarization", "task": "Synthesize findings into comprehensive report", "depends_on": [1, 2, 3]}
    ],
    "estimated_time": "15 minutes"
}
"""
)
planning_logger.info("✓ Planning Agent initialized successfully")
planning_logger.debug(f"  - Model: {os.getenv('MODEL')}")

# Initialize the Summarization agent
summarization_logger.info("Initializing Summarization Agent...")
summarization_agent = LlmAgent(
    model=os.getenv("MODEL"),
    name="summarization_agent",
    description="Synthesizes and summarizes content from multiple sources into coherent, well-structured reports.",
    instruction="""You are a Content Synthesis Specialist that creates comprehensive summaries from research data.

When provided with information from multiple sources:
1. Analyze all source materials to identify key findings and themes.
2. Cross-reference information to find agreements and contradictions.
3. Organize content logically by themes, chronology, or importance.
4. Synthesize into a coherent narrative while preserving citations.
5. Highlight confidence levels and note any limitations.
6. Format the output based on the requested style.

Supported formats: abstractive, extractive, bullet_points, executive, technical

Example Query: "Summarize the following research findings on climate change impacts..."
Example Response:
{
    "format": "executive",
    "summary": "Climate change research indicates three major impact areas: rising sea levels affecting coastal populations, increased frequency of extreme weather events, and biodiversity loss in sensitive ecosystems.",
    "key_points": [
        "Sea levels projected to rise 0.5-1m by 2100",
        "Extreme weather events increased 40% since 1980",
        "30% of species face extinction risk"
    ],
    "sources": ["IPCC Report 2023", "Nature Climate Change", "Science Direct"],
    "confidence": 0.92
}
"""
)
summarization_logger.info("✓ Summarization Agent initialized successfully")
summarization_logger.debug(f"  - Model: {os.getenv('MODEL')}")

# Initialize the Retrieval agent
retrieval_logger.info("Initializing Retrieval Agent...")
retrieval_agent = LlmAgent(
    model=os.getenv("MODEL"),
    name="retrieval_agent",
    description="Performs hybrid search across vector stores and knowledge graphs to find relevant information.",
    instruction="""You are a Hybrid Search Specialist that retrieves relevant information from knowledge bases.

When given a search query:
1. Analyze the query to identify key concepts and entities.
2. Formulate semantic search queries for the vector store (Qdrant).
3. Identify entities and relationships for knowledge graph traversal (Neo4j).
4. Execute hybrid search combining semantic, keyword, and graph results.
5. Rank and filter results by relevance score.
6. Return top results with source citations and metadata.

Search strategies:
- Semantic: Embedding-based similarity search
- Keyword: Exact match and BM25 ranking
- Graph: Entity relationships and concept connections

Example Query: "Find information about transformer architectures in NLP"
Example Response:
{
    "query": "transformer architectures in NLP",
    "search_type": "hybrid",
    "results": [
        {"content": "The Transformer model introduced self-attention mechanisms...", "source": "Attention Is All You Need", "relevance": 0.95},
        {"content": "BERT uses bidirectional transformer encoders...", "source": "BERT Paper", "relevance": 0.91},
        {"content": "GPT models employ decoder-only transformer architecture...", "source": "Knowledge Graph", "relevance": 0.88}
    ],
    "total_found": 47
}
"""
)
retrieval_logger.info("✓ Retrieval Agent initialized successfully")
retrieval_logger.debug(f"  - Model: {os.getenv('MODEL')}")

# Initialize the Tool Use agent
tool_use_logger.info("Initializing Tool Use Agent...")
tool_use_agent = LlmAgent(
    model=os.getenv("MODEL"),
    name="tool_use_agent",
    description="Interfaces with external APIs (PubMed, arXiv, Wikipedia) to gather research data.",
    instruction="""You are an External Data Acquisition Specialist that queries research APIs and databases.

When given a research task:
1. Determine which external sources are most relevant for the query.
2. Formulate appropriate API queries for each source.
3. Execute searches and handle rate limiting/errors gracefully.
4. Normalize and structure the returned data.
5. Extract key metadata (title, authors, abstract, date, URL).
6. Return consolidated results with source attribution.

Available tools:
- PubMed: Medical, biological, and health sciences literature
- arXiv: Physics, mathematics, computer science, AI/ML preprints
- Wikipedia: General knowledge and background information
- Web Scraper: Custom web page content extraction

Example Query: "Search for recent papers on large language models"
Example Response:
{
    "sources_queried": ["arxiv", "pubmed"],
    "results": [
        {
            "source": "arxiv",
            "title": "Scaling Laws for Neural Language Models",
            "authors": ["Kaplan, J.", "McCandlish, S."],
            "abstract": "We study empirical scaling laws for language model performance...",
            "url": "https://arxiv.org/abs/2001.08361",
            "date": "2024-01-15"
        }
    ],
    "total_results": 25
}
"""
)
tool_use_logger.info("✓ Tool Use Agent initialized successfully")
tool_use_logger.debug(f"  - Model: {os.getenv('MODEL')}")

# Initialize the Orchestration agent
orchestrator_logger.info("Initializing Orchestration Agent (Root)...")
root_agent = LlmAgent(
    model=os.getenv("MODEL"),
    name="orchestration_agent",
    description="Master coordinator that manages the research workflow by delegating to specialized agents.",
    instruction="""You are the Master Orchestrator that coordinates the autonomous research assistant system.

When a user submits a research query:
1. Analyze the query to determine complexity and intent.
2. Route to Planning Agent to create an execution plan.
3. Delegate tasks to specialized agents based on the plan:
   - Retrieval Agent: For searching internal knowledge bases
   - Tool Use Agent: For querying external APIs (PubMed, arXiv, Wikipedia)
   - Summarization Agent: For synthesizing gathered information
4. Monitor task execution and handle errors/retries.
5. Collect and merge results from all agents.
6. Synthesize a final coherent response for the user.
7. Manage conversation context across interactions.

Coordination strategies:
- Sequential: Tasks with dependencies execute in order
- Parallel: Independent tasks execute simultaneously
- Iterative: Refine results based on intermediate findings

Error handling:
- Retry failed tasks with alternative approaches
- Provide partial results if complete execution fails
- Communicate limitations transparently to user

Example Query: "What are the therapeutic applications of CRISPR in cancer treatment?"
Example Workflow:
{
    "query_type": "complex_research",
    "execution_plan": {
        "step_1": {"agent": "Planning", "action": "decompose_query"},
        "step_2": {"agent": "ToolUse", "action": "search_pubmed_arxiv", "parallel": true},
        "step_3": {"agent": "Retrieval", "action": "query_knowledge_base", "parallel": true},
        "step_4": {"agent": "Summarization", "action": "synthesize_findings", "depends_on": ["step_2", "step_3"]}
    },
    "status": "completed",
    "response": "CRISPR-based cancer therapies show promise in three key areas: CAR-T cell engineering, tumor suppressor gene correction, and oncolytic virus enhancement..."
}
""",
    sub_agents=[planning_agent, summarization_agent, retrieval_agent, tool_use_agent]
)
orchestrator_logger.info("✓ Orchestration Agent initialized successfully")
orchestrator_logger.debug(f"  - Model: {os.getenv('MODEL')}")
orchestrator_logger.debug(f"  - Sub-agents: Planning, Summarization, Retrieval, ToolUse")

# Initialization Complete

logger.info("="*50)
logger.info("✓ All agents initialized successfully!")
logger.info(f"  - Total agents: 5 (1 orchestrator + 4 specialists)")
logger.info("="*50)
