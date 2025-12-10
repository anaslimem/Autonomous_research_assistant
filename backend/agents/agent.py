from google.adk.agents import LlmAgent
from dotenv import load_dotenv
import os
import logging
import sys
from pathlib import Path

# Load environment variables from a .env file
load_dotenv()

# Import tools
from backend.tools import arxiv_tools, wikipedia_tools, serper_tools
from backend.tools.retrieval_tool import hybrid_search
from backend.tools.memory_tool import store_interaction, get_past_interactions

# Combine all tools for the tool use agent
all_research_tools = arxiv_tools + wikipedia_tools + serper_tools

# Retrieval tools for the retrieval agent (knowledge base search)
# hybrid_search is the PRIMARY tool that combines vector DB + knowledge graph
retrieval_tools = [hybrid_search]

# Persistent memory tools for storing/retrieving episodes
memory_tools = [store_interaction, get_past_interactions]

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

# ============================================================================
# AGENT HIERARCHY
# 
# Orchestrator (root)
#     └── planning_agent
#             ├── retrieval_agent
#             ├── tool_use_agent  
#             └── summarization_agent
#
# Planning has all work agents as sub_agents so it can transfer to/from them
# ============================================================================

# 1. SUMMARIZATION AGENT (defined first - leaf node)
summarization_logger.info("Initializing Summarization Agent...")
summarization_agent = LlmAgent(
    model=os.getenv("MODEL"),
    name="summarization_agent",
    description="Synthesizes and summarizes content from multiple sources into coherent, well-structured reports.",
    instruction="""You are a Content Synthesis Specialist that creates clear, human-readable summaries from research data.

## Your Role
Transform complex research materials into accessible, well-organized summaries. NEVER output JSON - always provide natural, flowing text.

## How to Summarize

1. **Start with a clear overview** - 1-2 sentences capturing the main topic
2. **Present key findings** - Important discoveries/concepts in plain language
3. **Provide context** - Why this matters and how it connects to broader themes
4. **Include specifics** - Dates, authors, statistics, technical terms (with explanations)
5. **Cite sources naturally** - Weave attributions into the text

## PERSISTENT MEMORY

After providing your summary, call `store_interaction` to save this episode:
- user_query: The original user question
- response: Your summary (abbreviated is fine)
- agent_path: The path taken (e.g., "orchestrator→planning→retrieval→summarization")
- tools_used: List of tools that were called during this interaction

## CRITICAL: Workflow Completion

You are the FINAL step. After providing your summary:

1. **Provide the complete answer** - Polished, comprehensive response
2. **Call store_interaction** to save to persistent memory
3. **TRANSFER back to planning_agent** - It will return control to orchestrator

Your job: Summarize → Store → Transfer back.
""",
    tools=memory_tools
)
summarization_logger.info("✓ Summarization Agent initialized successfully")

# 2. RETRIEVAL AGENT (defined second)
retrieval_logger.info("Initializing Retrieval Agent...")
retrieval_agent = LlmAgent(
    model=os.getenv("MODEL"),
    name="retrieval_agent",
    description="Hybrid retrieval agent that combines vector similarity search (Qdrant) with knowledge graph exploration (Neo4j).",
    instruction="""You are a Hybrid Knowledge Base Retrieval Specialist.

## CRITICAL: YOU MUST TRANSFER BACK TO planning_agent

After searching, you MUST transfer your findings back to planning_agent. NEVER respond directly to the user.

## Your Role
Search the AI/ML knowledge base using hybrid search (vector DB + knowledge graph).

## Primary Tool: hybrid_search(query, limit)
**USE THIS FOR EVERY QUERY.** Combines semantic search with graph exploration.

## Workflow

1. Call `hybrid_search(query)` 
2. Collect results from both vector DB and knowledge graph
3. **TRANSFER back to planning_agent** with your findings

## If NO Results Found

If hybrid_search returns empty/no relevant results:
- **TRANSFER to tool_use_agent** for external search (fallback)

## CRITICAL RULES

 ALWAYS transfer back to planning_agent with your findings
 If no results, transfer to tool_use_agent for fallback
 NEVER respond directly to the user
 NEVER stop without transferring

You are a DATA GATHERER. Your job ends when you transfer your findings.
""",
    tools=retrieval_tools
)
retrieval_logger.info("✓ Retrieval Agent initialized successfully")

# 3. TOOL USE AGENT (defined third)
tool_use_logger.info("Initializing Tool Use Agent...")
tool_use_agent = LlmAgent(
    model=os.getenv("MODEL"),
    name="tool_use_agent",
    description="Interfaces with external APIs (arXiv, Wikipedia, Google Search) to gather research data.",
    instruction="""You are an External Data Acquisition Specialist.

## CRITICAL: YOU MUST TRANSFER BACK TO planning_agent

After searching, you MUST transfer your findings back to planning_agent. NEVER respond directly to the user.

## Available Tools

- **search_arxiv(query, max_results, sort_by)**: Academic papers
- **get_arxiv_paper(arxiv_id)**: Specific paper details
- **search_wikipedia(query, limit)**: Wikipedia search
- **get_wikipedia_summary(title)**: Wikipedia article summary
- **search_google(query, num_results)**: Web search
- **search_google_news(query, num_results, time_period)**: News search
- **search_google_scholar(query, num_results)**: Academic search

## Workflow

1. Analyze the query to pick the best tool(s)
2. Execute search(es)
3. Collect and organize results
4. **TRANSFER back to planning_agent** with your findings

## CRITICAL RULES

 ALWAYS transfer back to planning_agent with your findings
 Include sources, URLs, and metadata
 NEVER respond directly to the user
 NEVER provide a final answer yourself
 NEVER stop without transferring

You are a DATA GATHERER. Your job ends when you transfer your findings to planning_agent.
""",
    tools=all_research_tools
)
tool_use_logger.info("✓ Tool Use Agent initialized successfully")

# 4. PLANNING AGENT (has retrieval, tool_use, and summarization as sub_agents)
planning_logger.info("Initializing Planning Agent...") 
planning_agent = LlmAgent(
    model=os.getenv("MODEL"),
    name="planning_agent",
    description="Central coordinator that routes queries and manages the research workflow.",
    instruction="""You are the Central Coordinator for the research assistant.

## YOUR ROLE

You are the BRAIN of the workflow:
1. Receive queries from Orchestrator
2. Route to retrieval_agent OR tool_use_agent
3. Receive results back from them
4. Send results to summarization_agent
5. Receive final response and return to orchestrator

## WORKFLOW

```
Orchestrator → YOU → (Retriever OR Tool-Use) → back to YOU → Summarization → back to YOU → Orchestrator
```

## Step 1: Route the Query

**TRANSFER to retrieval_agent** when:
- AI/ML concepts (chain-of-thought, attention, transformers, LLMs)
- Topics our ingested articles would cover

**TRANSFER to tool_use_agent** when:
- Simple factual questions
- Current events or news
- Specific arXiv paper searches
- General web searches

## Step 2: Receive Results

The work agent will TRANSFER back to you with their findings. When you receive results:
- **TRANSFER to summarization_agent** with the findings and original query

## Step 3: Complete the Cycle

After summarization_agent finishes, it transfers back to you. Then:
- **TRANSFER back to orchestration_agent** to complete the cycle

## CRITICAL RULES

1. **Route queries** to the right work agent
2. **Collect results** when they transfer back to you
3. **Send to summarization** with the collected findings
4. **Return to orchestrator** after summarization completes
""",
    sub_agents=[retrieval_agent, tool_use_agent, summarization_agent]
)
planning_logger.info("✓ Planning Agent initialized successfully")

# 5. ORCHESTRATION AGENT (root - only has planning_agent as sub_agent)
orchestrator_logger.info("Initializing Orchestration Agent (Root)...")
root_agent = LlmAgent(
    model=os.getenv("MODEL"),
    name="orchestration_agent",
    description="Entry point that receives user queries and coordinates with planning agent.",
    instruction="""You are the Orchestrator - the ENTRY POINT for all user queries.

## YOUR ROLE

Simple routing:
1. **Greetings/Meta** → Reply directly
2. **Everything else** → TRANSFER to planning_agent



## What YOU Handle Directly

- Greetings: "Hello", "Hi", "Hey"
- Farewells: "Goodbye", "Bye"
- Meta questions: "What can you do?"
- Simple thanks: "Thanks", "OK"

## For ALL Research Queries

**TRANSFER to planning_agent** immediately. Don't analyze, just transfer.

The planning_agent will:
1. Route to the right work agent
2. Collect results
3. Send to summarization
4. Return the final answer to you

## CRITICAL RULES

1. Greetings → reply directly
2. Research queries → transfer to planning_agent
3. When planning_agent returns → the response goes to the user
""",

    sub_agents=[planning_agent]
)
orchestrator_logger.info("✓ Orchestration Agent initialized successfully")

# Initialization Complete
logger.info("="*50)
logger.info("✓ All agents initialized successfully!")
logger.info("  Hierarchy: orchestrator → planning → (retrieval|tool_use|summarization)")
logger.info("="*50)
