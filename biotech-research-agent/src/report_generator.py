"""
report_generator.py
--------------------
Converts an AgentResult into a structured markdown research report.

Report sections:
  1. Title + metadata
  2. Research Question
  3. Background (gene/protein info from UniProt / NCBI Gene tools)
  4. Key Findings (PubMed evidence)
  5. Active Clinical Trials
  6. Drug/Compound Landscape
  7. Agent Reasoning Trace (tool calls, with iteration markers)
  8. Open Questions & Limitations
  9. References

Usage:
    from src.report_generator import generate_report, save_report

    result = agent.run("What phase 3 trials exist for KRAS G12C inhibitors?")
    md = generate_report(result)
    save_report(md, "outputs/kras_report.md")
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.agent import AgentResult


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def generate_report(result: AgentResult) -> str:
    """
    Convert an AgentResult into a structured markdown report.

    Parameters
    ----------
    result : AgentResult
        The output of BiotechResearchAgent.run().

    Returns
    -------
    str — full markdown document.
    """
    sections = [
        _title_block(result),
        _research_question(result),
        _answer_summary(result),
        _findings_by_tool(result),
        _reasoning_trace(result),
        _open_questions(result),
        _references(result),
        _footer(result),
    ]
    return "\n\n---\n\n".join(s for s in sections if s.strip())


def save_report(markdown: str, output_path: str) -> Path:
    """
    Save the markdown report to a file.

    Parameters
    ----------
    markdown : str
        The markdown content from generate_report().
    output_path : str
        File path to write to (will create parent dirs).

    Returns
    -------
    Path — the written file path.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")
    return path


# ------------------------------------------------------------------
# Section builders
# ------------------------------------------------------------------

def _title_block(result: AgentResult) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    n_calls = len(result.tool_calls)
    tools_used = list(dict.fromkeys(tc.tool_name for tc in result.tool_calls))
    tools_str = ", ".join(f"`{t}`" for t in tools_used) if tools_used else "none"

    return f"""# Biotech Research Report

| Field | Value |
|-------|-------|
| **Generated** | {ts} |
| **Model** | `{result.model}` |
| **Agent iterations** | {result.iterations} |
| **Tool calls** | {n_calls} ({tools_str}) |
| **Stop reason** | `{result.stopped_reason}` |"""


def _research_question(result: AgentResult) -> str:
    return f"""## Research Question

> {result.question}"""


def _answer_summary(result: AgentResult) -> str:
    answer = result.answer.strip()
    if not answer:
        answer = "_No answer generated._"
    return f"""## Summary

{answer}"""


def _findings_by_tool(result: AgentResult) -> str:
    """Group tool results by tool type and render themed sections."""
    if not result.tool_calls:
        return ""

    # Group calls by tool name
    by_tool: dict[str, list] = {}
    for tc in result.tool_calls:
        by_tool.setdefault(tc.tool_name, []).append(tc)

    parts = ["## Evidence Gathered"]

    # --- PubMed ---
    if "search_pubmed" in by_tool:
        parts.append("### Literature (PubMed)")
        for tc in by_tool["search_pubmed"]:
            query = tc.arguments.get("query", "")
            parts.append(f"**Query:** `{query}`\n")
            papers = _extract_list(tc.result, "papers")
            if papers:
                for p in papers[:5]:
                    title = p.get("title", "Untitled")
                    pmid = p.get("pmid", "")
                    year = p.get("year", "")
                    journal = p.get("journal", "")
                    abstract = p.get("abstract", "")[:300]
                    url = p.get("url", f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/")
                    parts.append(
                        f"- **[{title}]({url})**  \n"
                        f"  {journal} ({year}) · PMID: {pmid}  \n"
                        f"  _{abstract}..._\n"
                    )
            else:
                parts.append("_No papers returned._\n")

    # --- ClinicalTrials ---
    if "search_trials" in by_tool:
        parts.append("### Clinical Trials (ClinicalTrials.gov)")
        for tc in by_tool["search_trials"]:
            query = tc.arguments.get("query", "")
            parts.append(f"**Query:** `{query}`\n")
            trials = _extract_list(tc.result, "trials")
            if trials:
                for t in trials[:8]:
                    nct = t.get("nct_id", "N/A")
                    title = t.get("title", "Untitled")
                    phase = t.get("phase", "N/A")
                    status = t.get("status", "N/A")
                    conditions = ", ".join(t.get("conditions", []))
                    interventions = ", ".join(t.get("interventions", []))
                    url = t.get("url", f"https://clinicaltrials.gov/study/{nct}")
                    parts.append(
                        f"- **[{nct}]({url})** — {title}  \n"
                        f"  Phase: {phase} · Status: {status}  \n"
                        f"  Conditions: {conditions}  \n"
                        f"  Interventions: {interventions}\n"
                    )
            else:
                parts.append("_No trials returned._\n")

    # --- UniProt ---
    if "lookup_protein" in by_tool:
        parts.append("### Protein Information (UniProt)")
        for tc in by_tool["lookup_protein"]:
            query = tc.arguments.get("query", "")
            parts.append(f"**Query:** `{query}`\n")
            proteins = _extract_list(tc.result, "proteins")
            if proteins:
                for p in proteins[:3]:
                    acc = p.get("accession", "N/A")
                    name = p.get("protein_name", "N/A")
                    gene = p.get("gene_name", "N/A")
                    organism = p.get("organism", "N/A")
                    function = p.get("function", "")[:400]
                    url = p.get("url", f"https://www.uniprot.org/uniprot/{acc}")
                    diseases = p.get("disease_associations", [])
                    disease_str = "; ".join(diseases[:3]) if diseases else "None listed"
                    parts.append(
                        f"- **[{acc}]({url})** — {name} (gene: *{gene}*, {organism})  \n"
                        f"  Function: {function}  \n"
                        f"  Disease associations: {disease_str}\n"
                    )
            else:
                parts.append("_No proteins returned._\n")

    # --- Open Targets ---
    if "query_drug_targets" in by_tool:
        parts.append("### Drug–Target Associations (Open Targets)")
        for tc in by_tool["query_drug_targets"]:
            gene = tc.arguments.get("target_gene", "")
            parts.append(f"**Target gene:** `{gene}`\n")
            drugs = _extract_list(tc.result, "drugs")
            if drugs:
                for d in drugs[:8]:
                    drug_name = d.get("drug_name", "N/A")
                    phase = d.get("clinical_phase", "N/A")
                    moa = d.get("mechanism_of_action", "N/A")
                    diseases = ", ".join(d.get("diseases", [])[:3])
                    parts.append(
                        f"- **{drug_name}** · Phase {phase}  \n"
                        f"  MoA: {moa}  \n"
                        f"  Indications: {diseases}\n"
                    )
            else:
                parts.append("_No drug–target data returned._\n")

    # --- ChEMBL ---
    if "lookup_drug" in by_tool:
        parts.append("### Compound Data (ChEMBL)")
        for tc in by_tool["lookup_drug"]:
            query = tc.arguments.get("drug_name", "")
            parts.append(f"**Drug:** `{query}`\n")
            drugs = _extract_list(tc.result, "drugs")
            if drugs:
                for d in drugs[:3]:
                    chembl_id = d.get("chembl_id", "N/A")
                    name = d.get("name", "N/A")
                    mol_type = d.get("molecule_type", "N/A")
                    phase = d.get("max_clinical_phase", "N/A")
                    approved = "✅ Approved" if d.get("is_approved") else "🔬 Investigational"
                    url = d.get("url", f"https://www.ebi.ac.uk/chembl/compound_report_card/{chembl_id}/")
                    mechanisms = d.get("mechanisms_of_action", [])
                    moa_str = "; ".join(
                        m.get("mechanism", "") for m in mechanisms[:2] if m.get("mechanism")
                    ) or "Not recorded"
                    parts.append(
                        f"- **[{name}]({url})** (`{chembl_id}`) — {mol_type}  \n"
                        f"  Phase: {phase} · {approved}  \n"
                        f"  Mechanism: {moa_str}\n"
                    )
            else:
                parts.append("_No compound data returned._\n")

    # --- NCBI Gene ---
    if "search_genes" in by_tool:
        parts.append("### Gene Information (NCBI Gene)")
        for tc in by_tool["search_genes"]:
            query = tc.arguments.get("query", "")
            parts.append(f"**Query:** `{query}`\n")
            genes = _extract_list(tc.result, "genes")
            if genes:
                for g in genes[:3]:
                    symbol = g.get("symbol", "N/A")
                    full_name = g.get("full_name", "N/A")
                    gene_type = g.get("gene_type", "N/A")
                    location = g.get("chromosomal_location", "N/A")
                    summary = g.get("summary", "")[:400]
                    url = g.get("ncbi_url", "")
                    parts.append(
                        f"- **[{symbol}]({url})** — {full_name}  \n"
                        f"  Type: {gene_type} · Location: {location}  \n"
                        f"  _{summary}..._\n"
                    )
            else:
                parts.append("_No gene data returned._\n")

    return "\n\n".join(parts)


def _reasoning_trace(result: AgentResult) -> str:
    if not result.tool_calls:
        return ""

    lines = ["## Agent Reasoning Trace", ""]
    lines.append(
        "_This section shows every API call the agent made, in order. "
        "Useful for understanding how the agent built its answer._\n"
    )

    current_iter = None
    for i, tc in enumerate(result.tool_calls, 1):
        if tc.iteration != current_iter:
            current_iter = tc.iteration
            lines.append(f"### Iteration {current_iter}")

        # Format args
        args_str = ", ".join(f"`{k}={v!r}`" for k, v in tc.arguments.items())

        # Truncate result preview
        try:
            parsed = json.loads(tc.result)
            preview = json.dumps(parsed, indent=2)[:400]
        except Exception:
            preview = tc.result[:400]

        lines.append(
            f"**Step {i}** — `{tc.tool_name}({args_str})`\n"
            f"```json\n{preview}\n```\n"
        )

    return "\n".join(lines)


def _open_questions(result: AgentResult) -> str:
    caveats = []

    if result.stopped_reason == "max_iterations":
        caveats.append(
            "⚠️ The agent reached its iteration limit before completing its search. "
            "Some evidence may be missing — consider re-running with a higher `max_iterations`."
        )

    if result.stopped_reason == "error":
        caveats.append(
            "⚠️ The agent encountered an API error. Results may be incomplete."
        )

    # Check for tools that returned errors
    error_tools = []
    for tc in result.tool_calls:
        try:
            data = json.loads(tc.result)
            if "error" in data:
                error_tools.append(f"`{tc.tool_name}`")
        except Exception:
            pass

    if error_tools:
        caveats.append(
            f"⚠️ The following tools returned errors: {', '.join(error_tools)}. "
            "Results from those tools may be incomplete."
        )

    general = [
        "This report was generated by an automated AI agent and should be verified "
        "against primary sources before use in research or clinical decisions.",
        "Clinical trial status and drug approval information changes frequently — "
        "always check ClinicalTrials.gov and FDA/EMA databases directly for current status.",
        "PubMed abstracts are summaries only; full-text review is needed to assess methodology and validity.",
    ]

    all_items = caveats + general
    items_md = "\n".join(f"- {item}" for item in all_items)

    return f"""## Open Questions & Limitations

{items_md}"""


def _references(result: AgentResult) -> str:
    """Collect all URLs mentioned in tool results."""
    if not result.tool_calls:
        return ""

    refs = []
    seen = set()

    for tc in result.tool_calls:
        try:
            data = json.loads(tc.result)
            _collect_urls(data, refs, seen)
        except Exception:
            pass

    if not refs:
        return ""

    ref_lines = "\n".join(f"{i+1}. <{url}>" for i, url in enumerate(refs[:30]))
    return f"""## References

{ref_lines}"""


def _footer(result: AgentResult) -> str:
    return (
        "_Report generated by **Biotech Research Agent** — "
        "a ReAct agent using Groq + 6 free scientific APIs. "
        "[GitHub](https://github.com/yourusername/biotech-research-agent)_"
    )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _extract_list(result_json: str, key: str) -> list:
    """Safely extract a list from a JSON string result."""
    try:
        data = json.loads(result_json)
        return data.get(key, []) or []
    except Exception:
        return []


def _collect_urls(obj, refs: list, seen: set):
    """Recursively collect URL strings from a dict/list."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("url", "ncbi_url", "omim_url") and isinstance(v, str) and v not in seen:
                seen.add(v)
                refs.append(v)
            else:
                _collect_urls(v, refs, seen)
    elif isinstance(obj, list):
        for item in obj:
            _collect_urls(item, refs, seen)
