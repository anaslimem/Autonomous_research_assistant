from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from backend.api import app

client = TestClient(app)

def test_health_check():
    """Test the /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_root():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Research Assistant API", "status": "running"}

@patch("backend.api.get_runner")
@patch("backend.api.get_recent_episodes")
def test_chat_endpoint(mock_get_episodes, mock_get_runner):
    """Test the /chat endpoint with mocked runner."""
    # Mock dependencies
    mock_runner = MagicMock()
    mock_get_runner.return_value = mock_runner
    mock_get_episodes.return_value = []
    
    # Mock runner.run_async to yield a simple response
    async def mock_run_async(*args, **kwargs):
        event = MagicMock()
        event.author = "orchestrator"
        event.content.parts = [MagicMock(text="I am a test response")]
        yield event
        
    mock_runner.run_async = mock_run_async

    # Make request
    response = client.post("/chat", json={
        "message": "Hello test",
        "session_id": "test-session-123"
    })
    
    # Assertions
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "test-session-123"
    assert "I am a test response" in data["response"]

@patch("backend.api.get_all_episodes")
def test_list_sessions(mock_get_all):
    """Test the /sessions endpoint."""
    # Mock persistent storage return
    mock_episode = MagicMock()
    mock_episode.session_id = "sess-1"
    mock_episode.created_at.isoformat.return_value = "2023-01-01T12:00:00"
    mock_episode.user_query = "Test query"
    mock_get_all.return_value = [mock_episode]
    
    response = client.get("/sessions")
    
    assert response.status_code == 200
    sessions = response.json()
    assert len(sessions) == 1
    assert sessions[0]["session_id"] == "sess-1"
    assert sessions[0]["message_count"] == 1
