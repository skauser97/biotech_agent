"""
ArXiv API tool — search preprints, especially AI x biotech crossover.
Free, no API key required.
"""

import requests
from xml.etree import ElementTree as ET
from langchain_core.tools import tool


ARXIV_URL = "https://export.arxiv.org/api/query"
NS = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"


@tool
def search_arxiv(query: str, max_results: int = 5) -> str:
    """
    Search ArXiv for preprints. Best for cutting-edge AI + biology research:
    foundation models for biology, protein language models, GPU compute for biotech,
    autonomous labs, generative chemistry, AI-driven drug discovery, biomodels.
    Returns title, authors, abstract, and ArXiv link.

    Args:
        query: Search query (e.g. 'protein language model drug discovery', 'autonomous laboratory AI')
        max_results: Number of papers to return (default 5, max 10)
    """
    max_results = min(int(max_results), 10)  # coerce str→int in case model passes "5"

    params = {
        "search_query": f"all:{query}",
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    r = requests.get(ARXIV_URL, params=params, timeout=15)
    r.raise_for_status()

    root = ET.fromstring(r.text)
    total_el = root.find("{http://a9.com/-/spec/opensearch/1.1/}totalResults")
    total = total_el.text if total_el is not None else "?"
    entries = root.findall(f"{NS}entry")

    if not entries:
        return f"No ArXiv results found for: '{query}'"

    output = [f"**ArXiv Search: '{query}'** — {total} total results, showing {len(entries)}\n"]
    for i, entry in enumerate(entries, 1):
        title = entry.find(f"{NS}title")
        summary = entry.find(f"{NS}summary")
        published = entry.find(f"{NS}published")
        link = entry.find(f"{NS}id")
        authors = entry.findall(f"{NS}author")

        title_text = (title.text or "").replace("\n", " ").strip() if title is not None else "No title"
        abstract = (summary.text or "").replace("\n", " ").strip() if summary is not None else ""
        abstract = abstract[:500] + ("..." if len(abstract) > 500 else "")
        date = (published.text or "")[:10] if published is not None else "?"
        url = link.text.strip() if link is not None else "?"
        # Convert abstract URL to proper arxiv URL
        arxiv_id = url.replace("http://arxiv.org/abs/", "").replace("https://arxiv.org/abs/", "")
        author_list = ", ".join(
            (a.find(f"{NS}name").text or "") for a in authors[:3] if a.find(f"{NS}name") is not None
        )
        if len(authors) > 3:
            author_list += " et al."

        output.append(
            f"{i}. **{title_text}**\n"
            f"   {author_list} | {date}\n"
            f"   {abstract}\n"
            f"   🔗 https://arxiv.org/abs/{arxiv_id}\n"
        )
    return "\n".join(output)