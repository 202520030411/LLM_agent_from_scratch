import wikipediaapi

SCHEMA = {
    "type": "function",
    "function": {
        "name": "wikipedia_lookup",
        "description": "Look up a topic on Wikipedia and return a concise summary. Use this to get factual background information about people, places, events, or concepts.",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The topic or entity to look up on Wikipedia, e.g. 'Python programming language' or 'Albert Einstein'.",
                }
            },
            "required": ["topic"],
        },
    },
}

_wiki = wikipediaapi.Wikipedia(user_agent="LLMAgentFromScratch/1.0", language="en")


def run(topic: str) -> str:
    """Fetch a Wikipedia summary for the given topic."""
    page = _wiki.page(topic)
    if not page.exists():
        return f"No Wikipedia article found for '{topic}'. Try a different search term."
    # Return the first ~1500 chars of the summary to keep context manageable
    summary = page.summary[:1500]
    return f"**{page.title}**\n\n{summary}\n\n(Source: {page.fullurl})"
