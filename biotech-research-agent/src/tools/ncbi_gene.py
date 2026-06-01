"""
ncbi_gene.py
-------------
Search NCBI Gene database for gene information.

Returns gene summary, chromosomal location, pathways,
and links to related resources (OMIM, RefSeq).

Uses Bio.Entrez (Biopython) — same as PubMed, free.
"""

import os
import time
from loguru import logger
from Bio import Entrez


def _setup_entrez():
    Entrez.email = os.environ.get("NCBI_EMAIL", "research@example.com")


def search_genes(
    query: str,
    organism: str = "Homo sapiens",
    max_results: int = 3,
) -> dict:
    """
    Search NCBI Gene for genes matching a query.

    Parameters
    ----------
    query : str
        Gene symbol, name, or related term (e.g., "EGFR", "epidermal growth factor receptor").
    organism : str
        Filter by organism (default: "Homo sapiens").
    max_results : int

    Returns
    -------
    dict with keys: query, genes (list of dicts)
    """
    _setup_entrez()
    logger.debug(f"[ncbi_gene] Searching gene: {query!r}")

    search_term = f"{query}[Gene Name] AND {organism}[Organism]"

    try:
        # Step 1: Search for gene IDs
        handle = Entrez.esearch(db="gene", term=search_term, retmax=max_results)
        record = Entrez.read(handle)
        handle.close()

        id_list = record.get("IdList", [])
        if not id_list:
            # Try broader search
            handle = Entrez.esearch(
                db="gene", term=f"{query} AND {organism}[Organism]", retmax=max_results
            )
            record = Entrez.read(handle)
            handle.close()
            id_list = record.get("IdList", [])

        if not id_list:
            return {
                "query": query,
                "genes": [],
                "message": f"No gene found for '{query}' in {organism}.",
            }

        time.sleep(0.34)

        # Step 2: Fetch gene summaries
        handle = Entrez.esummary(db="gene", id=",".join(id_list))
        summaries = Entrez.read(handle)
        handle.close()

        genes = []
        doc_sum = summaries.get("DocumentSummarySet", {}).get("DocumentSummary", [])

        for entry in doc_sum:
            gene_id = str(entry.attributes.get("uid", "N/A"))
            name = entry.get("Name", "N/A")
            full_name = entry.get("Description", "N/A")
            summary = entry.get("Summary", "No summary available.")[:800]
            location = entry.get("MapLocation", "N/A")
            chromosome = entry.get("Chromosome", "N/A")
            gene_type = entry.get("GeneType", "N/A")
            organism_name = entry.get("Organism", {}).get("ScientificName", organism)

            # Other designations / aliases
            other_aliases = entry.get("OtherAliases", "")
            aliases = [a.strip() for a in other_aliases.split(",") if a.strip()][:5]

            # OMIM link
            omim_ids = []
            mim_list = entry.get("MIMLinks", {}).get("Link", [])
            if isinstance(mim_list, list):
                for mim in mim_list[:2]:
                    mim_id = mim.get("Id", "")
                    if mim_id:
                        omim_ids.append(str(mim_id))

            genes.append({
                "gene_id": gene_id,
                "symbol": name,
                "full_name": full_name,
                "gene_type": gene_type,
                "organism": organism_name,
                "chromosome": chromosome,
                "chromosomal_location": location,
                "aliases": aliases,
                "summary": summary,
                "omim_ids": omim_ids,
                "ncbi_url": f"https://www.ncbi.nlm.nih.gov/gene/{gene_id}",
                "omim_url": f"https://omim.org/entry/{omim_ids[0]}" if omim_ids else None,
            })

        time.sleep(0.34)
        logger.info(f"[ncbi_gene] Found {len(genes)} genes for {query!r}")

        return {
            "query": query,
            "organism": organism,
            "genes": genes,
        }

    except Exception as e:
        logger.error(f"[ncbi_gene] Error: {e}")
        return {
            "query": query,
            "genes": [],
            "error": str(e),
        }


# JSON schema for Groq tool calling
SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_genes",
        "description": (
            "Search NCBI Gene for gene information including chromosomal "
            "location, gene type, aliases, and a descriptive summary. "
            "Use this to get foundational gene information. Good for: "
            "'where is BRCA1 located?', 'what type of gene is TP53?', "
            "'what are the aliases for EGFR?', 'tell me about the KRAS gene'"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Gene symbol or name. Examples: "
                        "'KRAS', 'TP53', 'BRCA1', 'EGFR', 'ALK', 'MYC'"
                    ),
                },
                "organism": {
                    "type": "string",
                    "description": "Organism (default: 'Homo sapiens').",
                    "default": "Homo sapiens",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max genes to return (default 3).",
                    "default": 3,
                },
            },
            "required": ["query"],
        },
    },
}
