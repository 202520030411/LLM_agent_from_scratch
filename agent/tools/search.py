from ddgs import DDGS

SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for current information on a query and return the top results. Use this for recent news, facts you are unsure about, or anything that might not be in your training data.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up, e.g. 'latest AI research 2025'.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default 5, max 10).",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}


def run(query: str, max_results: int = 5) -> str:
    """Run a DuckDuckGo web search and return formatted results."""
    max_results = min(max_results, 10)
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return f"No results found for query: '{query}'"

        lines = [f"Search results for: '{query}'\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. **{r['title']}**")
            lines.append(f"   {r['href']}")
            lines.append(f"   {r['body']}\n")
        return "\n".join(lines)
    except Exception as e:
        return f"Search error: {e}"
