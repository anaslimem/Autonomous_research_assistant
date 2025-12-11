import streamlit as st
import requests
import uuid
import os

# Configuration
API_BASE = os.getenv("API_URL")

st.set_page_config(
    page_title="Research Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Agent Flow and Dark Mode Tweaks
st.markdown("""
<style>
    .agent-flow {
        font-family: 'Courier New', monospace;
        background-color: #1E1E2E;
        padding: 10px;
        border-radius: 8px;
        border-left: 4px solid #A855F7;
        margin-bottom: 10px;
        font-size: 0.9em;
        color: #CDD6F4;
    }
    .agent-step {
        display: inline-block;
        margin-right: 5px;
    }
    .agent-arrow {
        color: #6C7086;
        margin: 0 5px;
    }
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #11111B;
    }
    /* Chat message styling */
    .stChatMessage {
        background-color: #181825;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Session State Initialization
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- API Functions ---

def create_session():
    st.session_state.current_session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.rerun()

def get_sessions():
    try:
        response = requests.get(f"{API_BASE}/sessions")
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Failed to connect to backend: {e}")
    return []

def get_session_messages(session_id):
    try:
        response = requests.get(f"{API_BASE}/sessions/{session_id}/messages")
        if response.status_code == 200:
            return response.json().get("messages", [])
    except Exception:
        pass
    return []

def send_message(message):
    if not st.session_state.current_session_id:
        st.session_state.current_session_id = str(uuid.uuid4())
    
    payload = {
        "message": message,
        "session_id": st.session_state.current_session_id
    }
    
    try:
        response = requests.post(f"{API_BASE}/chat", json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.text}")
    except Exception as e:
        st.error(f"Connection error: {e}")
    return None

def delete_session(session_id):
    try:
        response = requests.delete(f"{API_BASE}/sessions/{session_id}")
        return response.status_code == 200
    except Exception as e:
        st.error(f"Failed to delete session: {e}")
    return False

# --- Sidebar ---

with st.sidebar:
    st.title("Research Agent")
    
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        if st.button("➕ New Chat", use_container_width=True):
            create_session()
    
    st.markdown("### Recent Conversations")
    sessions = get_sessions()
    
    for sess in sessions:
        sid = sess.get("session_id")
        label = sess.get("last_query", "New Chat")[:25] + "..."
        if not label: label = "New Chat"
        
        col_sess, col_del = st.columns([0.85, 0.15])
        
        with col_sess:
            # Highlight active session
            if sid == st.session_state.current_session_id:
                st.markdown(f"**👉 {label}**")
            else:
                if st.button(label, key=f"btn_{sid}", help=sid, use_container_width=True):
                    st.session_state.current_session_id = sid
                    st.session_state.messages = get_session_messages(sid)
                    st.rerun()
        
        with col_del:
            if st.button("X", key=f"del_{sid}", help="Delete chat"):
                if delete_session(sid):
                    # If deleted active session, reset state
                    if sid == st.session_state.current_session_id:
                        st.session_state.current_session_id = None
                        st.session_state.messages = []
                    st.rerun()

# --- Main Chat Area ---

st.title("Research Assistant")

if not st.session_state.current_session_id:
    st.info("Start a new chat or select one from the sidebar to begin.")
else:
    # Display message history
    # If we just switched sessions, messages might be loaded. 
    # If we are in a session, we should load history once or keep in sync?
    # For simplicity, we can fetch history on load if empty, or trust session_state
    
    # Sync with backend if empty (first load of session)
    if not st.session_state.messages and st.session_state.current_session_id:
        st.session_state.messages = get_session_messages(st.session_state.current_session_id)

    for msg in st.session_state.messages:
        # Normalize message format between local state and backend format
        # Backend returns: {user_query, agent_response, agent_path...}
        # We display User then Assistant
        
        # User Message
        with st.chat_message("user"):
            st.write(msg.get("user_query", ""))
        
        # Assistant Message
        with st.chat_message("assistant"):
            # Visualize Agent Flow
            path = msg.get("agent_path", [])
            # Handle if path is string "a -> b" or list
            if isinstance(path, str): 
                # Clean up string representation if it came from backend raw
                # backend persistent stores "planning->retrieval", api returns list usually?
                # looking at api.py, ChatResponse returns list[str], create_session messages returns list[str] in logic?
                # Actually api.py: get_session_messages returns what is stored in helper.
                # persistent.py stores string? No, mapped_column(String).
                # Wait, persistent.py has `agent_path: Mapped[str]`.
                # API ChatResponse has `agent_path: list[str]`.
                pass
                
            # If it's a list, join it. If string, use it.
            if isinstance(path, list):
                flow_html = " <span class='agent-arrow'>→</span> ".join([
                    f"<span class='agent-step'>{step.replace('_agent', '').capitalize()}</span>" 
                    for step in path
                ])
            else:
                flow_html = str(path)

            if flow_html:
                st.markdown(f"<div class='agent-flow'>{flow_html}</div>", unsafe_allow_html=True)
            
            st.write(msg.get("agent_response", ""))

    # Chat Input
    if prompt := st.chat_input("Ask about any research topic..."):
        # Display user message immediately
        with st.chat_message("user"):
            st.write(prompt)
        
        # Placeholder for assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response_data = send_message(prompt)
                
                if response_data:
                    # Parse response
                    answer = response_data.get("response")
                    path = response_data.get("agent_path", [])
                    
                    # Visualize Agent Flow
                    flow_html = " <span class='agent-arrow'>→</span> ".join([
                        f"<span class='agent-step'>{step.replace('_agent', '').capitalize()}</span>" 
                        for step in path
                    ])
                    st.markdown(f"<div class='agent-flow'>{flow_html}</div>", unsafe_allow_html=True)
                    st.write(answer)
                    
                    # Update session state with new message pair
                    new_msg = {
                        "user_query": prompt,
                        "agent_response": answer,
                        "agent_path": path
                    }
                    st.session_state.messages.append(new_msg)
