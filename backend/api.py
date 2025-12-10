from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uuid
import logging

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from contextlib import asynccontextmanager
from backend.agents.agent import root_agent
from backend.memory.persistent import (
    get_all_episodes, 
    get_recent_episodes, 
    delete_episodes_by_session,
    init_db
)

# Set up logging
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle events for the FastAPI application."""
    # Startup
    logger.info("Initializing database tables...")
    try:
        init_db()
        logger.info("Database tables initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        # Build might fail if DB isn't ready, but let's log it
    
    yield
    
    # Shutdown (if needed)
    logger.info("Shutting down API...")

# Initialize FastAPI app
app = FastAPI(
    title="Research Assistant API",
    description="API for the Autonomous Research Assistant",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session service for managing agent sessions
session_service = InMemorySessionService()

# Store active runners per session
active_runners: dict[str, Runner] = {}


# Request/Response models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    response: str
    agent_path: list[str]


class SessionInfo(BaseModel):
    session_id: str
    created_at: str
    message_count: int


# Helper to get runner for session
async def get_runner(session_id: str) -> Runner:
    # Ensure session exists in the service
    try:
        await session_service.get_session(session_id)
    except Exception:
        # Create new session if not found
        await session_service.create_session(
            session_id=session_id,
            app_name="research_assistant",
            user_id="user"
        )
        logger.info(f"Created new session: {session_id}")

    if session_id not in active_runners:
        active_runners[session_id] = Runner(
            agent=root_agent,
            app_name="research_assistant",
            session_service=session_service
        )
    return active_runners[session_id]


@app.get("/")
async def root():
    return {"message": "Research Assistant API", "status": "running"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message to the research assistant and get a response."""
    # Generate session ID if not provided
    session_id = request.session_id or str(uuid.uuid4())
    
    try:
        runner = await get_runner(session_id)
        
        # Debug: Check session history
        try:
            session = await session_service.get_session(session_id)
            history_len = len(session.state.get("messages", [])) if hasattr(session, "state") else "unknown"
            logger.info(f"Runner obtained for session {session_id}. Session object exists.")
        except Exception as e:
            logger.error(f"Error inspecting session: {e}")
        
        # Create user content
        user_content = types.Content(
            role="user",
            parts=[types.Part(text=request.message)]
        )
        
        # Run the agent and collect response
        agent_path = []
        final_response = ""
        
        async for event in runner.run_async(
            session_id=session_id,
            user_id="user",
            new_message=user_content
        ):
            # Track agent transfers
            if hasattr(event, 'author') and event.author:
                if event.author not in agent_path:
                    agent_path.append(event.author)
            
            # Collect text response
            if hasattr(event, 'content') and event.content:
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text:
                        final_response = part.text
        
        return ChatResponse(
            session_id=session_id,
            response=final_response,
            agent_path=agent_path
        )
    
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions")
async def list_sessions():
    """List all sessions with their episodes."""
    try:
        episodes = get_all_episodes(limit=100)
        
        # Group by session
        sessions = {}
        for ep in episodes:
            sid = ep.session_id
            if sid not in sessions:
                sessions[sid] = {
                    "session_id": sid,
                    "created_at": ep.created_at.isoformat() if ep.created_at else None,
                    "message_count": 0,
                    "last_query": ""
                }
            sessions[sid]["message_count"] += 1
            if not sessions[sid]["last_query"]:
                sessions[sid]["last_query"] = ep.user_query[:100]
        
        return list(sessions.values())
    
    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}/messages")
async def get_session_messages(session_id: str):
    """Get all messages for a specific session."""
    try:
        episodes = get_recent_episodes(session_id, limit=50)
        
        messages = []
        for ep in reversed(episodes):  # Chronological order
            messages.append({
                "id": str(ep.id),
                "user_query": ep.user_query,
                "agent_response": ep.agent_response,
                "agent_path": ep.agent_path,
                "tools_used": ep.tools_used,
                "feedback": ep.feedback,
                "created_at": ep.created_at.isoformat() if ep.created_at else None
            })
        
        return {"session_id": session_id, "messages": messages}
    
    except Exception as e:
        logger.error(f"Error getting messages: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and its runner."""
    # Delete from memory
    if session_id in active_runners:
        del active_runners[session_id]
        
    # Delete from persistent storage
    deleted = delete_episodes_by_session(session_id)
    
    if deleted:
        return {"message": f"Session {session_id} deleted"}
    else:
        # If not found in DB or error, but removed from memory, still return 200
        return {"message": f"Session {session_id} cleared from memory"}


# Health check
@app.get("/health")
async def health():
    return {"status": "healthy"}
