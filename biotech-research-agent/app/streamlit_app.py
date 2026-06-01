"""
app/streamlit_app.py
---------------------
Streamlit UI for the Biotech Research Agent.

Features:
  - Chat interface: ask any biotech/pharma question
  - Tool trace panel: see every API call the agent made (great for interviews)
  - Research report mode: download a structured markdown report
  - Example queries: one-click pre-filled questions
  - Session history: keeps the last 5 Q&A pairs

Run:
    streamlit run app/streamlit_app.py
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

# ------ path setup so we can import src/ from the project root ------
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.agent import BiotechResearchAgent, AgentResult
from src.report_generator import generate_report
from src.utils import format_tool_result_for_display

# -----------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="Biotech Research Agent",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------
# Custom CSS
# -----------------------------------------------------------------------
st.markdown(
    """
    <style>
    .tool-card {
        background: #f8f9fa;
        border-left: 4px solid #2196f3;
        border-radius: 4px;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-size: 0.85em;
    }
    .tool-name { font-weight: 700; color: #1565c0; }
    .tool-iter { color: #888; font-size: 0.8em; }
    .answer-box {
        background: #e8f5e9;
        border-left: 4px solid #43a047;
        border-radius: 4px;
        padding: 16px;
        margin-top: 10px;
    }
    .warn-box {
        background: #fff3e0;
        border-left: 4px solid #fb8c00;
        border-radius: 4px;
        padding: 12px;
        margin-top: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------
# Example queries
# -----------------------------------------------------------------------
EXAMPLE_QUERIES = [
    "What phase 3 trials exist for KRAS G12C inhibitors?",
    "What is the mechanism of action of osimertinib and what EGFR mutations does it target?",
    "What drugs are being developed for BRCA1-mutant cancers?",
    "Tell me about the PD-1/PD-L1 pathway and checkpoint inhibitors in lung cancer",
    "What is the chromosomal location of TP53 and what diseases is it linked to?",
    "What approved drugs target HER2? Include clinical trial landscape.",
    "Summarize pembrolizumab: mechanism, approvals, and active trials",
    "What are the latest CRISPR therapies in clinical trials?",
]

TOOL_ICONS = {
    "search_pubmed": "📄",
    "search_trials": "🏥",
    "lookup_protein": "🔬",
    "query_drug_targets": "🎯",
    "lookup_drug": "💊",
    "search_genes": "🧬",
}

# -----------------------------------------------------------------------
# Session state initialisation
# -----------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []   # list of AgentResult
if "agent" not in st.session_state:
    st.session_state.agent = None
if "running" not in st.session_state:
    st.session_state.running = False


# -----------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------
with st.sidebar:
    st.title("🧬 Biotech Research Agent")
    st.caption("ReAct agent · Groq · 6 free scientific APIs")

    st.divider()

    # API key
    api_key_input = st.text_input(
        "Groq API Key",
        type="password",
        value=os.environ.get("GROQ_API_KEY", ""),
        help="Get a free key at console.groq.com",
    )
    if api_key_input:
        os.environ["GROQ_API_KEY"] = api_key_input

    st.divider()

    # Agent config
    st.subheader("Agent Settings")
    model = st.selectbox(
        "Groq model",
        ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768", "gemma2-9b-it"],
        index=0,
    )
    max_iterations = st.slider("Max iterations", min_value=2, max_value=12, value=8)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.05)

    st.divider()

    # Example queries
    st.subheader("💡 Example Queries")
    for q in EXAMPLE_QUERIES:
        if st.button(q[:55] + ("…" if len(q) > 55 else ""), use_container_width=True):
            st.session_state["prefill_query"] = q

    st.divider()

    # Clear history
    if st.button("🗑 Clear history", use_container_width=True):
        st.session_state.history = []
        st.rerun()

    # About
    st.caption(
        "**Tools available:**  \n"
        "📄 PubMed · 🏥 ClinicalTrials  \n"
        "🔬 UniProt · 🎯 Open Targets  \n"
        "💊 ChEMBL · 🧬 NCBI Gene"
    )


# -----------------------------------------------------------------------
# Helper: build / reuse agent
# -----------------------------------------------------------------------
def get_agent(model: str, max_iterations: int, temperature: float) -> BiotechResearchAgent:
    if (
        st.session_state.agent is None
        or st.session_state.agent.model != model
        or st.session_state.agent.max_iterations != max_iterations
    ):
        st.session_state.agent = BiotechResearchAgent(
            model=model,
            max_iterations=max_iterations,
            temperature=temperature,
        )
    return st.session_state.agent


# -----------------------------------------------------------------------
# Helper: render a single AgentResult
# -----------------------------------------------------------------------
def render_result(result: AgentResult, index: int = 0, expanded: bool = True):
    ts = datetime.now().strftime("%H:%M")

    with st.container():
        # Header
        col1, col2 = st.columns([5, 1])
        with col1:
            st.markdown(f"### 🔍 {result.question[:120]}")
        with col2:
            st.caption(f"iter: {result.iterations} · calls: {len(result.tool_calls)}")

        # Answer
        stop_icon = "✅" if result.stopped_reason == "stop" else (
            "⚠️" if result.stopped_reason == "max_iterations" else "❌"
        )
        st.markdown(
            f'<div class="answer-box">{stop_icon} '
            f'{result.answer.replace(chr(10), "<br>")}</div>',
            unsafe_allow_html=True,
        )

        # Tabs: Tool Trace | Raw JSON | Report
        tab_trace, tab_json, tab_report = st.tabs(
            ["🔧 Tool Trace", "📋 Raw JSON", "📝 Research Report"]
        )

        with tab_trace:
            _render_tool_trace(result)

        with tab_json:
            st.json(result.to_dict())

        with tab_report:
            report_md = generate_report(result)
            st.markdown(report_md)
            st.download_button(
                label="⬇ Download Report (.md)",
                data=report_md,
                file_name=f"biotech_report_{index}.md",
                mime="text/markdown",
                key=f"dl_{index}_{id(result)}",
            )


def _render_tool_trace(result: AgentResult):
    if not result.tool_calls:
        st.info("No tool calls — the agent answered from prior knowledge.")
        return

    st.caption(
        f"The agent made **{len(result.tool_calls)} API calls** across "
        f"**{result.iterations} reasoning iterations**."
    )

    current_iter = None
    for i, tc in enumerate(result.tool_calls, 1):
        if tc.iteration != current_iter:
            current_iter = tc.iteration
            st.markdown(f"**— Iteration {current_iter} —**")

        icon = TOOL_ICONS.get(tc.tool_name, "🔧")
        args_str = ", ".join(f"{k}={v!r}" for k, v in tc.arguments.items())

        with st.expander(
            f"{icon} Step {i}: `{tc.tool_name}({args_str[:80]})`",
            expanded=(i == 1),
        ):
            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.markdown("**Arguments**")
                st.json(tc.arguments)
            with col_b:
                st.markdown("**Result preview**")
                formatted = format_tool_result_for_display(tc.tool_name, tc.result)
                st.code(formatted[:1200], language="json")


# -----------------------------------------------------------------------
# Main area
# -----------------------------------------------------------------------
st.title("🧬 Biotech Research Agent")
st.markdown(
    "Ask any question about drugs, genes, proteins, or clinical trials. "
    "The agent searches **6 free scientific databases** and synthesises an answer."
)

# Query input
prefill = st.session_state.pop("prefill_query", "")
query = st.text_area(
    "Your research question",
    value=prefill,
    height=90,
    placeholder="e.g. What phase 3 trials exist for KRAS G12C inhibitors?",
)

col_run, col_space = st.columns([2, 5])
with col_run:
    run_button = st.button(
        "🔬 Run Research",
        disabled=st.session_state.running,
        type="primary",
        use_container_width=True,
    )

# Run agent
if run_button and query.strip():
    if not os.environ.get("GROQ_API_KEY"):
        st.error("⚠️ Please enter your Groq API key in the sidebar.")
    else:
        st.session_state.running = True

        with st.spinner("Agent is researching… (this may take 20–60 s)"):
            try:
                agent = get_agent(model, max_iterations, temperature)
                t0 = time.time()
                result = agent.run(query.strip())
                elapsed = time.time() - t0
                result._elapsed = round(elapsed, 1)
            except Exception as e:
                st.error(f"Agent error: {e}")
                st.session_state.running = False
                st.stop()

        # Prepend to history (newest first)
        st.session_state.history.insert(0, result)
        if len(st.session_state.history) > 5:
            st.session_state.history = st.session_state.history[:5]

        st.session_state.running = False

elif run_button and not query.strip():
    st.warning("Please enter a research question.")

# Display history
if st.session_state.history:
    for i, result in enumerate(st.session_state.history):
        render_result(result, index=i, expanded=(i == 0))
        if i < len(st.session_state.history) - 1:
            st.divider()
else:
    # Welcome / instructions
    st.info(
        "👈 Enter a question above or pick an example from the sidebar to get started.\n\n"
        "**What this agent can answer:**\n"
        "- Drug mechanisms, approvals, and clinical stages\n"
        "- Active clinical trials by indication or drug\n"
        "- Gene function, location, and disease links\n"
        "- Protein structure, pathways, and UniProt data\n"
        "- Drug–target associations from Open Targets\n"
        "- Literature search from PubMed\n\n"
        "**After each answer** you'll see:\n"
        "- 🔧 **Tool Trace** — every API call the agent made\n"
        "- 📋 **Raw JSON** — full structured output\n"
        "- 📝 **Research Report** — formatted markdown you can download"
    )
