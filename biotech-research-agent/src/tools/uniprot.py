"""
uniprot.py
-----------
Look up protein and gene information from UniProt.

UniProt is the gold standard for protein sequence, function,
and disease association data. Completely free REST API.

API docs: https://www.uniprot.org/help/api
"""

import time
import requests
from loguru import logger

BASE_URL = "https://rest.uniprot.org/uniprotkb"
HEADERS = {"Accept": "application/json"}


def lookup_protein(
    query: str,
    organism: str = "Homo sapiens",
    max_results: int = 3,
) -> dict:
    """
    Search UniProt for proteins matching a gene name or protein description.

    Parameters
    ----------
    query : str
        Gene name or protein name (e.g., "EGFR", "BRCA2", "PD-L1", "ACE2").
    organism : str
        Filter by organism (default: "Homo sapiens").
    max_results : int
        Max proteins to return.

    Returns
    -------
    dict with keys: query, proteins (list of dicts)
    """
    logger.debug(f"[uniprot] Looking up: {query!r}")

    search_query = f"({query}) AND (organism_name:{organism}) AND (reviewed:true)"

    params = {
        "query": search_query,
        "format": "json",
        "size": max_results,
        "fields": (
            "accession,id,gene_names,protein_name,organism_name,"
            "cc_function,cc_disease,cc_pathway,cc_subcellular_location,"
            "xref_pdb,length,sequence_version"
        ),
    }

    try:
        response = requests.get(
            f"{BASE_URL}/search",
            params=params,
            headers=HEADERS,
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()

        results = data.get("results", [])
        proteins = []

        for entry in results:
            accession = entry.get("primaryAccession", "N/A")
            entry_name = entry.get("uniProtkbId", "N/A")

            # Gene names
            genes = entry.get("genes", [])
            gene_names = []
            for g in genes:
                primary = g.get("geneName", {}).get("value")
                if primary:
                    gene_names.append(primary)

            # Protein name
            protein_desc = entry.get("proteinDescription", {})
            rec_name = protein_desc.get("recommendedName", {})
            protein_name = rec_name.get("fullName", {}).get("value", "N/A")

            # Function
            comments = entry.get("comments", [])
            function_text = ""
            disease_associations = []
            pathways = []
            location = ""

            for comment in comments:
                comment_type = comment.get("commentType", "")

                if comment_type == "FUNCTION":
                    texts = comment.get("texts", [])
                    if texts:
                        function_text = texts[0].get("value", "")[:800]

                elif comment_type == "DISEASE":
                    disease = comment.get("disease", {})
                    disease_name = disease.get("diseaseId", "")
                    if disease_name:
                        disease_associations.append(disease_name)

                elif comment_type == "PATHWAY":
                    texts = comment.get("texts", [])
                    for t in texts:
                        pathways.append(t.get("value", "")[:200])

                elif comment_type == "SUBCELLULAR LOCATION":
                    locs = comment.get("subcellularLocations", [])
                    location_names = [
                        loc.get("location", {}).get("value", "")
                        for loc in locs
                        if loc.get("location", {}).get("value")
                    ]
                    location = ", ".join(location_names[:3])

            organism_name = entry.get("organism", {}).get("scientificName", "N/A")
            length = entry.get("sequence", {}).get("length", "N/A")

            proteins.append({
                "accession": accession,
                "entry_name": entry_name,
                "gene_names": gene_names,
                "protein_name": protein_name,
                "organism": organism_name,
                "function": function_text,
                "disease_associations": disease_associations[:5],
                "pathways": pathways[:3],
                "subcellular_location": location,
                "sequence_length": length,
                "url": f"https://www.uniprot.org/uniprotkb/{accession}",
            })

        time.sleep(0.5)
        logger.info(f"[uniprot] Found {len(proteins)} proteins for {query!r}")

        return {
            "query": query,
            "organism_filter": organism,
            "proteins": proteins,
        }

    except Exception as e:
        logger.error(f"[uniprot] Error: {e}")
        return {
            "query": query,
            "proteins": [],
            "error": str(e),
        }


# JSON schema for Groq tool calling
SCHEMA = {
    "type": "function",
    "function": {
        "name": "lookup_protein",
        "description": (
            "Look up protein or gene information in UniProt. Returns protein "
            "function, disease associations, pathways, and subcellular location. "
            "Use this to understand what a gene/protein does, what diseases it's "
            "linked to, and where it's found in the cell. Good for: 'what does "
            "EGFR do?', 'what diseases is BRCA2 associated with?', "
            "'what pathway is KRAS in?'"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Gene symbol or protein name. Examples: "
                        "'EGFR', 'BRCA2', 'PD-L1', 'ACE2', 'TP53', 'HER2'"
                    ),
                },
                "organism": {
                    "type": "string",
                    "description": "Organism filter (default: 'Homo sapiens').",
                    "default": "Homo sapiens",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max proteins to return (default 3).",
                    "default": 3,
                },
            },
            "required": ["query"],
        },
    },
}
