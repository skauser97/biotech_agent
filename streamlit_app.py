"""
Biotech Research Assistant — Streamlit UI
Run: streamlit run streamlit_app.py
"""

import os
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Load .env from the same directory as this file
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Biotech Research Assistant",
    page_icon="🧬",
    layout="wide",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp {
        max-width: 900px;
        margin: 0 auto;
    }
    .tool-badge {
        display: inline-block;
        background: #1a1a2e;
        color: #00d4aa;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        margin-right: 4px;
        margin-bottom: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ── Session State ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8]
if "agent" not in st.session_state:
    st.session_state.agent = None


# ── Build Agent (cached) ─────────────────────────────────────────────────────
@st.cache_resource
def get_agent():
    from agent import build_agent
    return build_agent()


# ── Header ───────────────────────────────────────────────────────────────────
st.title("🧬 Biotech Research Assistant")
st.caption("AI-powered research across PubMed, ArXiv, ClinicalTrials.gov, OpenAlex & the web")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    backend = os.getenv("LLM_BACKEND", "groq").upper()
    model = os.getenv("GROQ_MODEL" if backend == "GROQ" else "OLLAMA_MODEL", "unknown")
    st.info(f"**Backend:** {backend}\n\n**Model:** `{model}`")

    st.divider()
    st.subheader("Available Tools")
    tools_info = {
        "search_pubmed": "PubMed — biomedical papers",
        "search_arxiv": "ArXiv — AI/bio preprints",
        "search_clinical_trials": "ClinicalTrials.gov — drug trials",
        "search_openalex": "OpenAlex — broad research",
        "search_web": "DuckDuckGo — company news",
        "search_web_news": "DuckDuckGo News — breaking news",
    }
    for tool, desc in tools_info.items():
        st.markdown(f"- **{tool}** — {desc}")

    st.divider()
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        st.rerun()

    st.divider()
    st.subheader("Example Queries")
    examples = [
        "Latest papers on protein language models for drug discovery",
        "What CRISPR trials are recruiting for sickle cell disease?",
        "Overview of Recursion Pharmaceuticals' pipeline",
        "Compare Isomorphic Labs and Insilico Medicine",
        "Recent FDA approvals for gene therapy",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=f"ex_{ex[:20]}"):
            st.session_state.pending_query = ex
            st.rerun()


# ── Load Agent ───────────────────────────────────────────────────────────────
try:
    agent = get_agent()
except Exception as e:
    st.error(f"Failed to load agent: {e}\n\nCheck your `.env` file.")
    st.stop()


# ── Display Chat History ─────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("tools_used"):
            tools_html = " ".join(
                f'<span class="tool-badge">{t}</span>' for t in msg["tools_used"]
            )
            st.markdown(f"Tools used: {tools_html}", unsafe_allow_html=True)


# ── Handle Input ─────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask a biotech research question...")

# Check for pending query from sidebar examples
if "pending_query" in st.session_state:
    user_input = st.session_state.pending_query
    del st.session_state.pending_query

if user_input:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Researching..."):
            try:
                from langchain_core.messages import HumanMessage

                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                input_msg = {"messages": [HumanMessage(content=user_input)]}

                tool_calls_made = []
                final_response = ""

                for chunk in agent.stream(input_msg, config=config, stream_mode="updates"):
                    if "tools" in chunk:
                        for msg in chunk["tools"].get("messages", []):
                            tool_name = getattr(msg, "name", "")
                            if tool_name and tool_name not in tool_calls_made:
                                tool_calls_made.append(tool_name)

                    if "agent" in chunk:
                        for msg in chunk["agent"].get("messages", []):
                            content = getattr(msg, "content", "")
                            if content and not getattr(msg, "tool_calls", []):
                                final_response = content

                st.markdown(final_response)

                if tool_calls_made:
                    tools_html = " ".join(
                        f'<span class="tool-badge">{t}</span>' for t in tool_calls_made
                    )
                    st.markdown(f"Tools used: {tools_html}", unsafe_allow_html=True)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_response,
                    "tools_used": tool_calls_made,
                })

            except Exception as e:
                err = str(e)
                if "tool_use_failed" in err or "Failed to call a function" in err:
                    st.error(
                        "Tool calling failed — the model produced a malformed response. "
                        "This happens intermittently. Please try again."
                    )
                else:
                    st.error(f"Error: {err[:500]}")
