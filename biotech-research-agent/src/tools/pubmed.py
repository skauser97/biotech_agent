"""
pubmed.py
----------
Search PubMed for scientific papers via NCBI Entrez API.

Free, no API key required — just provide an email address in .env
so NCBI can contact you if you accidentally overload their servers.

Rate limit: 3 requests/second without API key, 10/s with one.
We default to 0.34s delay between calls (< 3/s).

Entrez docs: https://www.ncbi.nlm.nih.gov/books/NBK25499/
"""

import os
import time
from typing import Optional
from loguru import logger

from Bio import Entrez


def _setup_entrez():
    """Configure Entrez with the user's email from .env."""
    email = os.environ.get("NCBI_EMAIL", "research@example.com")
    Entrez.email = email


def search_pubmed(
    query: str,
    max_results: int = 8,
    sort: str = "relevance",
) -> dict:
    """
    Search PubMed and return paper metadata + abstracts.

    Parameters
    ----------
    query : str
        PubMed search query. Supports field tags e.g.
        "KRAS[Gene] AND pancreatic cancer[Disease] AND clinical trial[PT]"
    max_results : int
        Maximum number of papers to return.
    sort : str
        "relevance" | "pub_date"

    Returns
    -------
    dict with keys:
        query, total_found, papers (list of dicts)
    """
    _setup_entrust()
    logger.debug(f"[pubmed] Searching: {query!r} (max={max_results})")

    try:
        # Step 1: Search for IDs
        search_handle = Entrez.esearch(
            db="pubmed",
            term=query,
            retmax=max_results,
            sort=sort,
        )
        search_results = Entrez.read(search_handle)
        search_handle.close()

        id_list = search_results.get("IdList", [])
        total_found = int(search_results.get("Count", 0))

        if not id_list:
            return {
                "query": query,
                "total_found": 0,
                "papers": [],
                "message": "No papers found for this query.",
            }

        time.sleep(0.34)  # NCBI rate limit

        # Step 2: Fetch details for each paper
        fetch_handle = Entrez.efetch(
            db="pubmed",
            id=",".join(id_list),
            rettype="abstract",
            retmode="xml",
        )
        records = Entrez.read(fetch_handle)
        fetch_handle.close()

        papers = []
        for record in records.get("PubmedArticle", []):
            try:
                article = record["MedlineCitation"]["Article"]
                pmid = str(record["MedlineCitation"]["PMID"])

                title = str(article.get("ArticleTitle", "No title"))

                # Abstract
                abstract_text = ""
                abstract = article.get("Abstract", {})
                if abstract:
                    abstract_parts = abstract.get("AbstractText", [])
                    if isinstance(abstract_parts, list):
                        abstract_text = " ".join(str(p) for p in abstract_parts)
                    else:
                        abstract_text = str(abstract_parts)

                # Authors
                author_list = article.get("AuthorList", [])
                authors = []
                for author in author_list[:3]:  # first 3 authors
                    if "LastName" in author:
                        name = author["LastName"]
                        if "ForeName" in author:
                            name += f" {author['ForeName'][0]}"
                        authors.append(name)
                if len(author_list) > 3:
                    authors.append("et al.")

                # Publication year
                pub_date = article.get("Journal", {}).get(
                    "JournalIssue", {}
                ).get("PubDate", {})
                year = pub_date.get("Year", pub_date.get("MedlineDate", "N/A"))

                # Journal
                journal = article.get("Journal", {}).get("Title", "Unknown Journal")

                papers.append({
                    "pmid": pmid,
                    "title": title,
                    "authors": ", ".join(authors),
                    "journal": journal,
                    "year": str(year),
                    "abstract": abstract_text[:1000] if abstract_text else "No abstract available.",
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                })

            except Exception as e:
                logger.debug(f"[pubmed] Error parsing record: {e}")
                continue

        logger.info(f"[pubmed] Found {total_found} papers, returning {len(papers)}")
        return {
            "query": query,
            "total_found": total_found,
            "papers": papers,
        }

    except Exception as e:
        logger.error(f"[pubmed] Search error: {e}")
        return {
            "query": query,
            "total_found": 0,
            "papers": [],
            "error": str(e),
        }


def _setup_entrust():
    """Alias to fix a typo-safe call."""
    _setup_entrez()


# JSON schema for Groq tool calling
SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_pubmed",
        "description": (
            "Search PubMed for peer-reviewed scientific papers. Use this to find "
            "evidence from the literature about drugs, diseases, genes, mechanisms, "
            "clinical trials, and biomarkers. Returns paper titles, authors, abstracts, "
            "and PubMed links. Good for: 'what does the literature say about X?', "
            "'find papers on Y mechanism', 'evidence for Z in cancer'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "PubMed search query. Use specific terms. Can use field tags: "
                        "[Title], [MeSH], [Gene], [Drug]. Examples: "
                        "'KRAS G12C inhibitor lung cancer', "
                        "'PD-1 immunotherapy clinical trial review[Title]'"
                    ),
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of papers to return (default 5, max 15).",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}
