
import pytest
from unittest.mock import MagicMock, patch
from backend.memory.persistent import store_episode, get_recent_episodes, delete_episodes_by_session, Episode

@patch("backend.memory.persistent.SessionLocal")
def test_store_episode(mock_session_cls):
    """Test storing an episode in the database."""
    # Setup mock session
    mock_session = MagicMock()
    mock_session_cls.return_value.__enter__.return_value = mock_session
    
    # Call function
    episode = store_episode(
        session_id="sess-1",
        user_query="Hello",
        agent_response="Hi there",
        agent_path="orchestrator",
        tools_used=[]
    )
    
    # Verify DB interactions
    mock_session.add.assert_called_once()
    mock_session.commit.assert_called_once()
    mock_session.refresh.assert_called_once()
    
    # Verify object attributes
    assert episode.session_id == "sess-1"
    assert episode.user_query == "Hello"
    assert episode.agent_response == "Hi there"

@patch("backend.memory.persistent.SessionLocal")
def test_get_recent_episodes(mock_session_cls):
    """Test retrieving recent episodes."""
    mock_session = MagicMock()
    mock_session_cls.return_value.__enter__.return_value = mock_session
    
    # Mock query chain
    mock_query = mock_session.query.return_value
    mock_filter = mock_query.filter.return_value
    mock_order = mock_filter.order_by.return_value
    mock_limit = mock_order.limit.return_value
    
    # Return fake data
    fake_episodes = [Episode(session_id="sess-1", user_query="Q1"), Episode(session_id="sess-1", user_query="Q2")]
    mock_limit.all.return_value = fake_episodes
    
    episodes = get_recent_episodes("sess-1", limit=2)
    
    assert len(episodes) == 2
    assert episodes[0].user_query == "Q1"

@patch("backend.memory.persistent.SessionLocal")
def test_delete_episodes_by_session(mock_session_cls):
    """Test deleting episodes."""
    mock_session = MagicMock()
    mock_session_cls.return_value.__enter__.return_value = mock_session
    
    result = delete_episodes_by_session("sess-1")
    
    assert result is True
    # Verify delete was called
    mock_session.query(Episode).filter().delete.assert_called_once()
    mock_session.commit.assert_called_once()
