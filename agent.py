"""
Biotech Research Agent — LangGraph ReAct agent with Ollama/Groq backend.
"""

import os
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent

# Load environment
load_dotenv()

# ── Tools ────────────────────────────────────────────────────────────────────
from tools.pubmed import search_pubmed
from tools.arxiv import search_arxiv
from tools.clinical_trials import search_clinical_trials
from tools.openalex import search_openalex
from tools.web_search import search_web, search_web_news

TOOLS = [
    search_pubmed,
    search_arxiv,
    search_clinical_trials,
    search_openalex,
    search_web,
    search_web_news,
]

# ── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a world-class biotech research assistant with deep expertise in:
- AI × Biology: protein language models, foundation models for genomics, AlphaFold, biomodels
- Autonomous labs: self-driving laboratories, lab robotics, closed-loop experimentation
- GPU compute for science: accelerated molecular dynamics, ML training infrastructure
- Drug discovery: AI-driven pipelines, generative chemistry, target identification
- Biotech companies: Recursion, Isomorphic Labs, Insilico Medicine, Ginkgo Bioworks, etc.
- Clinical development: trial phases, FDA approvals, drug pipelines
- Genomics & CRISPR: gene editing, synthetic biology

Your behavior:
1. ALWAYS use tools to retrieve real, current information — never rely solely on your training data for specific papers, trials, or company news.
2. For academic questions → use search_pubmed and/or search_arxiv (prefer arxiv for AI×bio crossover).
3. For drug/trial questions → use search_clinical_trials.
4. For company news, funding, recent events → use search_web_news or search_web.
5. For broad landscape questions → combine multiple tools.
6. Synthesize results into clear, insightful summaries. Highlight key findings, trends, and implications.
7. Always include source links so the user can dig deeper.
8. When you're unsure whether something is recent enough, use web search to check.
9. Be concise but substantive. Researchers want signal, not noise.
10. If asked to compare or analyze, give your own synthesis after presenting the data.

CRITICAL TOOL USE RULES:
- If the user asks you to summarize, explain, or analyze content that was ALREADY returned earlier in this conversation (e.g. "summarize the above", "explain those results", "what do those articles say"), answer DIRECTLY from the conversation context. Do NOT call any tools.
- Only call tools when you need NEW information not already present in the conversation.
- Never call a tool with an empty query string.
- Always pass max_results as an integer (e.g. 5), never as a string (e.g. "5").

Format your responses clearly with paper titles bolded, links on their own line, and a brief "Key Takeaway" at the end when synthesizing multiple sources.
"""

# ── Groq Fallback Models ─────────────────────────────────────────────────────
# Ordered by preference. If one fails tool calling, the next is tried.
GROQ_FALLBACK_MODELS = [
    "meta-llama/llama-4-scout-17b-16e-instruct",  # best, newest
    "qwen/qwen3-32b",                              # strong alternative
    "llama-3.1-8b-instant",                         # fastest, most reliable
]


# ── LLM Factory ───────────────────────────────────────────────────────────────
def get_llm(model_override=None):
    """Return configured LLM based on LLM_BACKEND env var."""
    backend = os.getenv("LLM_BACKEND", "groq").lower()

    if backend == "groq":
        from langchain_groq import ChatGroq
        model = model_override or os.getenv("GROQ_MODEL", GROQ_FALLBACK_MODELS[0])
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in .env")
        print(f"  Using Groq → {model}")
        return ChatGroq(model=model, api_key=api_key, temperature=0)

    elif backend == "ollama":
        from langchain_ollama import ChatOllama
        model = model_override or os.getenv("OLLAMA_MODEL", "qwen2.5:14b")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        print(f"  Using Ollama → {model} @ {base_url}")
        return ChatOllama(model=model, base_url=base_url, temperature=0)

    else:
        raise ValueError(f"Unknown LLM_BACKEND: '{backend}'. Choose 'groq' or 'ollama'.")


# ── Agent Builder ─────────────────────────────────────────────────────────────
def build_agent(model_override=None):
    """Build and return the LangGraph ReAct agent."""
    llm = get_llm(model_override)
    agent = create_react_agent(
        model=llm,
        tools=TOOLS,
        prompt=SystemMessage(content=SYSTEM_PROMPT),
    )
    return agent


def get_fallback_models():
    """Return the list of fallback models for the current backend."""
    backend = os.getenv("LLM_BACKEND", "groq").lower()
    if backend == "groq":
        return GROQ_FALLBACK_MODELS
    return []


# ── Single-query with fallback ───────────────────────────────────────────────
def run_query_with_fallback(query: str, thread_id: str = "default"):
    """Run a query with automatic model fallback on tool-calling failures.
    Returns (response_text, model_used, tools_used).
    """
    from langchain_core.messages import HumanMessage

    models = get_fallback_models()
    primary_model = os.getenv("GROQ_MODEL", models[0] if models else None)

    # Build ordered list: primary first, then fallbacks (skip duplicates)
    model_queue = [primary_model] if primary_model else []
    for m in models:
        if m not in model_queue:
            model_queue.append(m)

    last_error = None
    for model in model_queue:
        try:
            agent = build_agent(model_override=model)
            config = {"configurable": {"thread_id": thread_id}}
            input_msg = {"messages": [HumanMessage(content=query)]}

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

            return final_response, model, tool_calls_made

        except Exception as e:
            err = str(e)
            if "tool_use_failed" in err or "Failed to call a function" in err:
                print(f"  ⚠ {model} failed tool calling, trying next model...")
                last_error = e
                continue
            else:
                raise  # Non-tool-calling error, don't retry

    raise last_error  # All models failed


# ── Simple single-query helper (no fallback) ─────────────────────────────────
def run_query(agent, query: str, thread_id: str = "default") -> str:
    """Run a single query and return the final response text."""
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        config=config,
    )
    # Extract last AI message
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if hasattr(msg, "content") and msg.__class__.__name__ in ("AIMessage",):
            return msg.content
    return str(result)