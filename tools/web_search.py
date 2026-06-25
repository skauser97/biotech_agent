"""
DuckDuckGo web search tool — news, company info, recent developments.
Free, no API key required.
"""

from langchain_core.tools import tool


@tool
def search_web(query: str, max_results: int = 5) -> str:
    """
    Search the web for recent news, company information, funding rounds,
    product launches, and anything not covered by academic databases.
    Use for: biotech company news, GPU/compute announcements, drug approvals,
    regulatory updates, AI lab news (Recursion, Isomorphic, Insilico, etc.),
    startup funding, recent events (last few weeks/months).

    Args:
        query: Search query (e.g. 'Recursion Pharmaceuticals 2025 pipeline update')
        max_results: Number of results (default 5, max 10)
    """
    from ddgs import DDGS

    max_results = min(int(max_results), 10)  # coerce str→int in case model passes "5"

    try:
        ddgs = DDGS()
        results = list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        return f"Web search failed: {e}"

    if not results:
        return f"No web results found for: '{query}'"

    output = [f"**Web Search: '{query}'** — {len(results)} results\n"]
    for i, r in enumerate(results, 1):
        title = r.get("title", "No title")
        body = r.get("body", "")[:300] + ("..." if len(r.get("body", "")) > 300 else "")
        url = r.get("href", "")
        output.append(
            f"{i}. **{title}**\n"
            f"   {body}\n"
            f"   🔗 {url}\n"
        )
    return "\n".join(output)


@tool
def search_web_news(query: str, max_results: int = 5) -> str:
    """
    Search for recent news articles specifically.
    Use when the user asks about 'latest', 'recent', 'this week', 'news about' something.

    Args:
        query: News search query (e.g. 'AI biotech funding 2025')
        max_results: Number of news items (default 5)
    """
    from ddgs import DDGS

    max_results = min(int(max_results), 10)  # coerce str→int in case model passes "5"

    try:
        ddgs = DDGS()
        results = list(ddgs.news(query, max_results=max_results))
    except Exception as e:
        return f"News search failed: {e}"

    if not results:
        return f"No news found for: '{query}'"

    output = [f"**News Search: '{query}'** — {len(results)} articles\n"]
    for i, r in enumerate(results, 1):
        title = r.get("title", "No title")
        body = r.get("body", "")[:300] + ("..." if len(r.get("body", "")) > 300 else "")
        url = r.get("url", "")
        date = r.get("date", "?")
        source = r.get("source", "?")
        output.append(
            f"{i}. **{title}** [{source}, {date}]\n"
            f"   {body}\n"
            f"   🔗 {url}\n"
        )
    return "\n".join(output)