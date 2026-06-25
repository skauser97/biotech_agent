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


# ── Validate config on startup ───────────────────────────────────────────────
def check_config():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key and os.getenv("LLM_BACKEND", "groq").lower() == "groq":
        return False, "GROQ_API_KEY not set. Check your .env file or Streamlit secrets."
    return True, ""


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


# ── Validate Config ──────────────────────────────────────────────────────────
ok, err_msg = check_config()
if not ok:
    st.error(err_msg)
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

    # Get agent response with fallback
    with st.chat_message("assistant"):
        with st.spinner("Researching..."):
            try:
                from agent import run_query_with_fallback

                response, model_used, tool_calls_made = run_query_with_fallback(
                    user_input, st.session_state.thread_id
                )

                st.markdown(response)

                # Show model and tools used
                meta_parts = []
                if tool_calls_made:
                    tools_html = " ".join(
                        f'<span class="tool-badge">{t}</span>' for t in tool_calls_made
                    )
                    meta_parts.append(f"Tools: {tools_html}")
                meta_parts.append(f'<span style="color: #888; font-size: 0.75rem;">Model: {model_used}</span>')
                st.markdown(" &nbsp;|&nbsp; ".join(meta_parts), unsafe_allow_html=True)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "tools_used": tool_calls_made,
                })

            except Exception as e:
                err = str(e)
                if "tool_use_failed" in err or "Failed to call a function" in err:
                    st.error(
                        "All models failed tool calling. Please try rephrasing your question."
                    )
                else:
                    st.error(f"Error: {err[:500]}")
