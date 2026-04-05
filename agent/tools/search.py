import os
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for current, up-to-date information on a query and return the top results. Use this for recent news, current events, or anything that may have changed recently.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query, e.g. 'US China tariffs 2026 latest news'.",
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

_client = None


def _get_client() -> TavilyClient:
    global _client
    if _client is None:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise EnvironmentError("TAVILY_API_KEY is not set in .env")
        _client = TavilyClient(api_key=api_key)
    return _client


def run(query: str, max_results: int = 5) -> str:
    """Run a Tavily web search and return formatted results."""
    max_results = min(max_results, 10)
    try:
        response = _get_client().search(
            query=query,
            max_results=max_results,
            search_depth="basic",
            include_answer=True,
        )

        lines = [f"Search results for: '{query}'\n"]

        if response.get("answer"):
            lines.append(f"**Summary:** {response['answer']}\n")

        for i, r in enumerate(response.get("results", []), 1):
            published = f" ({r['published_date']})" if r.get("published_date") else ""
            lines.append(f"{i}. **{r['title']}**{published}")
            lines.append(f"   {r['url']}")
            lines.append(f"   {r['content'][:300]}\n")

        return "\n".join(lines)
    except Exception as e:
        return f"Search error: {e}"
