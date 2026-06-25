"""
OpenAlex API tool — open scholarly research database.
Free, no API key required. Good for broad research landscape queries.
"""

import requests
from langchain_core.tools import tool


BASE_URL = "https://api.openalex.org"


@tool
def search_openalex(query: str, max_results: int = 5, from_year: int = 2020) -> str:
    """
    Search OpenAlex for scientific papers across all disciplines.
    Good for: research landscape overview, finding influential papers,
    autonomous labs, GPU compute in science, AI models for biology,
    cross-disciplinary topics not well-covered by PubMed.
    Returns title, year, citation count, journal, and link.

    Args:
        query: Search terms (e.g. 'self-driving laboratory automation', 'GPU accelerated molecular dynamics')
        max_results: Number of results (default 5, max 10)
        from_year: Only include papers from this year onwards (default 2020)
    """
    max_results = min(int(max_results), 10)  # coerce str→int in case model passes "5"

    params = {
        "search": query,
        "per-page": max_results,
        "sort": "cited_by_count:desc",
        "filter": f"publication_year:>{from_year - 1}",
        "select": "title,publication_year,cited_by_count,primary_location,authorships,doi,open_access",
        "mailto": "research@biotech-agent.local",  # polite pool access
    }

    r = requests.get(f"{BASE_URL}/works", params=params, timeout=15)
    r.raise_for_status()
    data = r.json()

    results = data.get("results", [])
    meta = data.get("meta", {})
    total = meta.get("count", 0)

    if not results:
        return f"No OpenAlex results found for: '{query}' (from {from_year})"

    output = [f"**OpenAlex Search: '{query}'** — {total} total results (since {from_year}), showing {len(results)}\n"]
    for i, paper in enumerate(results, 1):
        title = paper.get("title") or "No title"
        year = paper.get("publication_year", "?")
        citations = paper.get("cited_by_count", 0)
        doi = paper.get("doi", "")
        oa = paper.get("open_access", {})
        oa_url = oa.get("oa_url", "")

        # Journal
        loc = paper.get("primary_location") or {}
        source = loc.get("source") or {}
        journal = source.get("display_name", "Unknown journal")

        # Authors
        authorships = paper.get("authorships", [])[:3]
        authors = ", ".join(
            a.get("author", {}).get("display_name", "") for a in authorships
        )
        if len(paper.get("authorships", [])) > 3:
            authors += " et al."

        url = oa_url or (f"https://doi.org/{doi.replace('https://doi.org/', '')}" if doi else "")

        output.append(
            f"{i}. **{title}**\n"
            f"   {authors} | {journal} ({year}) | Cited: {citations}\n"
            f"   🔗 {url or 'No link available'}\n"
        )
    return "\n".join(output)