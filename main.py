#!/usr/bin/env python3
"""
Biotech Research Assistant — CLI entry point.
Run: python main.py
"""

import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.spinner import Spinner
from rich.live import Live
from rich.text import Text
from rich import print as rprint

# ── Bootstrap ─────────────────────────────────────────────────────────────────
# Load .env from the same directory as this file
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

console = Console()

BANNER = """
╔══════════════════════════════════════════════════════╗
║     🧬  Biotech Research Assistant  🤖              ║
║     AI × Biology × Drugs × Autonomous Labs          ║
╚══════════════════════════════════════════════════════╝
"""

HELP_TEXT = """
**Commands:**
- Type any research question to get started
- `tools`  — list available search tools
- `clear`  — clear conversation history
- `switch` — switch between Groq / Ollama
- `help`   — show this message
- `quit` / `exit` / Ctrl+C — exit

**Example queries:**
- *What are the latest papers on protein language models for drug discovery?*
- *Give me an overview of Recursion Pharmaceuticals' current pipeline*
- *What CRISPR trials are currently recruiting for sickle cell disease?*
- *Summarize the state of autonomous laboratories in 2025*
- *Compare Isomorphic Labs and Insilico Medicine approaches*
"""

TOOLS_TEXT = """
| Tool | Source | Best for |
|------|--------|----------|
| search_pubmed | NCBI PubMed | Peer-reviewed biomedical papers |
| search_arxiv | ArXiv | AI × bio preprints, GPU compute |
| search_clinical_trials | ClinicalTrials.gov | Drug trials, pipeline status |
| search_openalex | OpenAlex | Broad research landscape |
| search_web | DuckDuckGo | Company news, recent events |
| search_web_news | DuckDuckGo News | Breaking news, funding rounds |
"""


def print_banner():
    console.print(BANNER, style="bold cyan")
    backend = os.getenv("LLM_BACKEND", "groq").upper()
    model = os.getenv("GROQ_MODEL" if backend == "GROQ" else "OLLAMA_MODEL", "?")
    console.print(f"  Backend: [bold green]{backend}[/] → [dim]{model}[/]")
    console.print(f"  Type [bold]help[/] to see commands\n")


def stream_response(agent, query: str, thread_id: str):
    """Stream the agent response with a live spinner while tools run."""
    from langchain_core.messages import HumanMessage

    config = {"configurable": {"thread_id": thread_id}}
    input_msg = {"messages": [HumanMessage(content=query)]}

    tool_calls_made = []
    final_response = ""

    with Live(console=console, refresh_per_second=10) as live:
        live.update(Text("🔍 Thinking...", style="dim"))

        for chunk in agent.stream(input_msg, config=config, stream_mode="updates"):
            # Track tool usage
            if "tools" in chunk:
                for msg in chunk["tools"].get("messages", []):
                    tool_name = getattr(msg, "name", "")
                    if tool_name and tool_name not in tool_calls_made:
                        tool_calls_made.append(tool_name)
                        live.update(Text(f"🔧 Using: {', '.join(tool_calls_made)}...", style="dim yellow"))

            # Capture final agent message
            if "agent" in chunk:
                for msg in chunk["agent"].get("messages", []):
                    content = getattr(msg, "content", "")
                    if content and not getattr(msg, "tool_calls", []):
                        final_response = content

        live.update(Text(""))

    return final_response, tool_calls_made


def main():
    print_banner()

    # Build agent
    console.print("[dim]Loading agent...[/]")
    try:
        from agent import build_agent
        agent = build_agent()
        console.print("[green]✓ Agent ready[/]\n")
    except Exception as e:
        console.print(f"[red]✗ Failed to load agent: {e}[/]")
        console.print("[yellow]Tip: Check your .env file. Copy .env.example → .env and fill in your API key.[/]")
        sys.exit(1)

    thread_id = str(uuid.uuid4())[:8]
    console.print(Rule(f"[dim]Session {thread_id}[/]"))

    while True:
        try:
            query = Prompt.ask("\n[bold cyan]You[/]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye! 🧬[/]")
            break

        if not query:
            continue

        # Built-in commands
        if query.lower() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye! 🧬[/]")
            break

        elif query.lower() == "help":
            console.print(Markdown(HELP_TEXT))
            continue

        elif query.lower() == "tools":
            console.print(Markdown(TOOLS_TEXT))
            continue

        elif query.lower() == "clear":
            thread_id = str(uuid.uuid4())[:8]
            console.clear()
            print_banner()
            console.print(Rule(f"[dim]New session {thread_id}[/]"))
            continue

        elif query.lower() == "switch":
            current = os.getenv("LLM_BACKEND", "groq")
            new_backend = "ollama" if current == "groq" else "groq"
            os.environ["LLM_BACKEND"] = new_backend
            console.print(f"[yellow]Switching to {new_backend.upper()}... rebuilding agent[/]")
            try:
                from agent import build_agent
                agent = build_agent()
                thread_id = str(uuid.uuid4())[:8]
                console.print(f"[green]✓ Switched to {new_backend.upper()}[/]")
            except Exception as e:
                console.print(f"[red]✗ Switch failed: {e}[/]")
                os.environ["LLM_BACKEND"] = current
            continue

        # Run query
        console.print()
        try:
            response, tools_used = stream_response(agent, query, thread_id)

            if tools_used:
                console.print(f"[dim]Tools used: {', '.join(tools_used)}[/]\n")

            console.print(Panel(
                Markdown(response),
                title="[bold green]Assistant[/]",
                border_style="green",
                padding=(1, 2),
            ))

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted.[/]")
        except Exception as e:
            err_str = str(e)
            # Friendly messages for common failures
            if "tool_use_failed" in err_str or "Failed to call a function" in err_str:
                console.print(Panel(
                    "[red]Tool calling failed.[/]\n\n"
                    "The model produced a malformed tool call. This can happen intermittently.\n"
                    "[bold]Try your question again[/] — it often works on retry.\n\n"
                    "If it keeps failing, try a different model in [bold].env[/]:\n\n"
                    "  [green]GROQ_MODEL=meta-llama/llama-4-scout-17b-16e-instruct[/]  (recommended)\n"
                    "  [green]GROQ_MODEL=qwen/qwen3-32b[/]  (alternative)\n"
                    "  [green]GROQ_MODEL=llama-3.1-8b-instant[/]  (fastest)",
                    title="[red]Model Error[/]", border_style="red"
                ))
            elif "api_key" in err_str.lower() or "authentication" in err_str.lower() or "401" in err_str:
                console.print(Panel(
                    "[red]API key error.[/]\n\nCheck that GROQ_API_KEY is set correctly in your [bold].env[/] file.",
                    title="[red]Auth Error[/]", border_style="red"
                ))
            elif "connection" in err_str.lower() or "ollama" in err_str.lower():
                console.print(Panel(
                    "[red]Cannot reach Ollama.[/]\n\nMake sure Ollama is running:\n\n  [green]ollama serve[/]",
                    title="[red]Connection Error[/]", border_style="red"
                ))
            else:
                console.print(f"[red]Error:[/] {err_str[:300]}")


if __name__ == "__main__":
    main()