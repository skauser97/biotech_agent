"""
PubMed E-utilities tool — search biomedical literature.
Free, no API key required.
"""

import requests
from langchain_core.tools import tool


BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def _fetch_abstracts(pmids: list[str]) -> list[dict]:
    """Fetch title + abstract for a list of PubMed IDs."""
    if not pmids:
        return []
    ids = ",".join(pmids)
    url = f"{BASE_URL}/efetch.fcgi"
    params = {"db": "pubmed", "id": ids, "retmode": "xml", "rettype": "abstract"}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()

    # Quick XML parse — no external lib needed
    from xml.etree import ElementTree as ET
    root = ET.fromstring(r.text)
    results = []
    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        title_el = article.find(".//ArticleTitle")
        abstract_el = article.find(".//AbstractText")
        year_el = article.find(".//PubDate/Year")
        journal_el = article.find(".//Journal/Title")
        authors = article.findall(".//Author/LastName")

        pmid = pmid_el.text if pmid_el is not None else "?"
        title = title_el.text if title_el is not None else "No title"
        abstract = abstract_el.text if abstract_el is not None else "No abstract available"
        year = year_el.text if year_el is not None else "?"
        journal = journal_el.text if journal_el is not None else "?"
        author_list = ", ".join(a.text for a in authors[:3] if a.text)
        if len(authors) > 3:
            author_list += " et al."

        results.append({
            "pmid": pmid,
            "title": title,
            "authors": author_list,
            "year": year,
            "journal": journal,
            "abstract": (abstract or "")[:600] + ("..." if abstract and len(abstract) > 600 else ""),
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        })
    return results


@tool
def search_pubmed(query: str, max_results: int = 5) -> str:
    """
    Search PubMed for biomedical research papers.
    Use for: peer-reviewed papers on drugs, diseases, genomics, clinical studies,
    molecular biology, biotech, and anything in the biomedical literature.
    Returns title, authors, year, journal, abstract snippet, and PubMed URL.

    Args:
        query: Search terms (e.g. 'CRISPR cancer therapy 2024', 'AlphaFold protein structure')
        max_results: Number of papers to return (default 5, max 10)
    """
    max_results = min(int(max_results), 10)  # coerce str→int in case model passes "5"

    # Step 1: search for IDs
    search_url = f"{BASE_URL}/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance",
    }
    r = requests.get(search_url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    pmids = data["esearchresult"]["idlist"]
    total = data["esearchresult"]["count"]

    if not pmids:
        return f"No PubMed results found for: '{query}'"

    # Step 2: fetch abstracts
    papers = _fetch_abstracts(pmids)

    output = [f"**PubMed Search: '{query}'** — {total} total results, showing {len(papers)}\n"]
    for i, p in enumerate(papers, 1):
        output.append(
            f"{i}. **{p['title']}**\n"
            f"   {p['authors']} | {p['journal']} ({p['year']})\n"
            f"   {p['abstract']}\n"
            f"   🔗 {p['url']}\n"
        )
    return "\n".join(output)