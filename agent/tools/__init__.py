from agent.tools import calculator, wikipedia, search

# All tools available to the agent
ALL_TOOLS = [calculator, wikipedia, search]

# OpenAI-format schemas to pass in the API request
TOOL_SCHEMAS = [t.SCHEMA for t in ALL_TOOLS]

# Dispatch map: tool name -> run function
TOOL_REGISTRY: dict = {
    "calculator": calculator.run,
    "wikipedia_lookup": wikipedia.run,
    "web_search": search.run,
}


def dispatch(name: str, arguments: dict) -> str:
    """Call the named tool with the given arguments and return its string output."""
    func = TOOL_REGISTRY.get(name)
    if func is None:
        return f"Unknown tool: '{name}'"
    return func(**arguments)
