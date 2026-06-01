"""
tools/__init__.py
------------------
Central tool registry.

Provides:
  - TOOL_SCHEMAS   : list of JSON schemas for Groq tool calling
  - TOOL_REGISTRY  : maps tool name → callable function
  - call_tool()    : dispatches a tool call by name
"""

import json
from loguru import logger

from src.tools.pubmed import search_pubmed, SCHEMA as PUBMED_SCHEMA
from src.tools.clinical_trials import search_trials, SCHEMA as TRIALS_SCHEMA
from src.tools.uniprot import lookup_protein, SCHEMA as UNIPROT_SCHEMA
from src.tools.opentargets import query_drug_targets, SCHEMA as OPENTARGETS_SCHEMA
from src.tools.chembl import lookup_drug, SCHEMA as CHEMBL_SCHEMA
from src.tools.ncbi_gene import search_genes, SCHEMA as GENE_SCHEMA


# All schemas passed to Groq — tells the LLM what tools exist and when to use them
TOOL_SCHEMAS = [
    PUBMED_SCHEMA,
    TRIALS_SCHEMA,
    UNIPROT_SCHEMA,
    OPENTARGETS_SCHEMA,
    CHEMBL_SCHEMA,
    GENE_SCHEMA,
]

# Maps tool function name → Python callable
TOOL_REGISTRY = {
    "search_pubmed": search_pubmed,
    "search_trials": search_trials,
    "lookup_protein": lookup_protein,
    "query_drug_targets": query_drug_targets,
    "lookup_drug": lookup_drug,
    "search_genes": search_genes,
}


def call_tool(tool_name: str, tool_args: dict) -> str:
    """
    Execute a tool by name and return a JSON string result.

    This is called by the agent loop whenever Groq requests a tool call.

    Parameters
    ----------
    tool_name : str
        Must match a key in TOOL_REGISTRY.
    tool_args : dict
        Arguments parsed from the LLM's tool call.

    Returns
    -------
    str — JSON-encoded result (passed back to the LLM as a tool message).
    """
    if tool_name not in TOOL_REGISTRY:
        error = {"error": f"Unknown tool: '{tool_name}'. Available: {list(TOOL_REGISTRY.keys())}"}
        return json.dumps(error)

    logger.info(f"[tools] Calling: {tool_name}({', '.join(f'{k}={v!r}' for k,v in tool_args.items())})")

    try:
        fn = TOOL_REGISTRY[tool_name]
        result = fn(**tool_args)
        return json.dumps(result, default=str)
    except TypeError as e:
        error = {
            "error": f"Invalid arguments for {tool_name}: {e}",
            "tip": "Check the tool schema for required parameters.",
        }
        logger.warning(f"[tools] TypeError in {tool_name}: {e}")
        return json.dumps(error)
    except Exception as e:
        error = {"error": f"Tool execution error in {tool_name}: {e}"}
        logger.error(f"[tools] Error in {tool_name}: {e}")
        return json.dumps(error)
