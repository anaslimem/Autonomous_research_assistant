from backend.memory.persistent import store_episode, get_recent_episodes
from google.adk.tools import ToolContext
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def store_interaction(
    tool_context: ToolContext,
    user_query: str,
    response: str,
    agent_path: str,
    tools_used: Optional[list[str]] = None
) -> str:
    """
    Store a completed interaction in persistent memory.
    Called by summarization agent after providing a response.
    
    Args:
        tool_context: ADK context (provides session ID automatically)
        user_query: The original user question
        response: The final response provided
        agent_path: Path taken (e.g., "planning→retrieval→summarization")
        tools_used: List of tools that were called
    
    Returns:
        Confirmation message
    """
    try:
        # Access session ID via private invocation context as it's not exposed publicly in ToolContext
        session_id = tool_context._invocation_context.session.id
        episode = store_episode(
            session_id=session_id,
            user_query=user_query,
            agent_response=response,
            agent_path=agent_path,
            tools_used=tools_used or []
        )
        logger.info(f"Stored interaction for session {session_id}")
        return f"Interaction stored successfully (episode {episode.id})"
    except Exception as e:
        logger.error(f"Failed to store interaction: {e}")
        return f"Failed to store interaction: {e}"


def get_past_interactions(
    tool_context: ToolContext,
    limit: int = 5
) -> str:
    """
    Retrieve past interactions from persistent memory for context.
    
    Args:
        tool_context: ADK context (provides session ID automatically)
        limit: Number of past episodes to retrieve
    
    Returns:
        Formatted string of past interactions
    """
    try:
        # Access session ID via private invocation context
        session_id = tool_context._invocation_context.session.id
        episodes = get_recent_episodes(session_id, limit)
        
        if not episodes:
            return "No past interactions found for this session."
        
        output = f"Found {len(episodes)} past interactions:\n\n"
        for i, ep in enumerate(episodes, 1):
            output += f"[{i}] Query: {ep.user_query[:100]}...\n"
            output += f"    Path: {ep.agent_path}\n"
            output += f"    Feedback: {ep.feedback or 'None'}\n\n"
        
        return output
    except Exception as e:
        logger.error(f"Failed to retrieve past interactions: {e}")
        return f"Failed to retrieve past interactions: {e}"