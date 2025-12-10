import uuid
import os
import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import create_engine, Text, String, DateTime, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from sqlalchemy.orm import Mapped, mapped_column, sessionmaker, DeclarativeBase
from dotenv import load_dotenv

# Set up logging
logger = logging.getLogger(__name__)

load_dotenv()

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise EnvironmentError("DATABASE_URL environment variable is required")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
logger.info(f"Connected to PostgreSQL database")


# Base class for models
class Base(DeclarativeBase):
    pass


class Episode(Base):
    """Stores a complete interaction episode for persistent memory."""
    __tablename__ = "episodes"
    
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    user_query: Mapped[str] = mapped_column(Text, nullable=False)
    agent_response: Mapped[str] = mapped_column(Text, nullable=False)
    agent_path: Mapped[str] = mapped_column(String(255), nullable=False)
    tools_used: Mapped[list] = mapped_column(JSON, default=list)
    feedback: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


def init_db():
    """Create all tables in the database."""
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully")


def store_episode(
    session_id: str,
    user_query: str,
    agent_response: str,
    agent_path: str,
    tools_used: list = None
) -> Episode:
    """Store a new episode in persistent memory."""
    with SessionLocal() as session:
        episode = Episode(
            session_id=session_id,
            user_query=user_query,
            agent_response=agent_response,
            agent_path=agent_path,
            tools_used=tools_used or []
        )
        session.add(episode)
        session.commit()
        session.refresh(episode)
        logger.info(f"Stored episode {episode.id} for session {session_id}")
        return episode


def get_recent_episodes(session_id: str, limit: int = 10) -> list[Episode]:
    """Get recent episodes for a specific session."""
    with SessionLocal() as session:
        episodes = session.query(Episode).filter(
            Episode.session_id == session_id
        ).order_by(Episode.created_at.desc()).limit(limit).all()
        return episodes


def update_feedback(episode_id: str, feedback: str) -> bool:
    """Update the feedback for an episode."""
    with SessionLocal() as session:
        episode = session.query(Episode).filter(Episode.id == episode_id).first()
        if episode:
            episode.feedback = feedback
            session.commit()
            logger.info(f"Updated feedback for episode {episode_id}: {feedback}")
            return True
        return False


def get_all_episodes(limit: int = 100) -> list[Episode]:
    """Get all episodes across all sessions."""
    with SessionLocal() as session:
        episodes = session.query(Episode).order_by(
            Episode.created_at.desc()
        ).limit(limit).all()
        return episodes


def delete_episodes_by_session(session_id: str) -> bool:
    """Delete all episodes for a specific session."""
    try:
        with SessionLocal() as session:
            session.query(Episode).filter(Episode.session_id == session_id).delete()
            session.commit()
            logger.info(f"Deleted all episodes for session {session_id}")
            return True
    except Exception as e:
        logger.error(f"Failed to delete episodes for session {session_id}: {e}")
        return False