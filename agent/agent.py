import json
from openai import BadRequestError
from agent.llm import chat_completion, extract_tool_calls, build_tool_result_message
from agent.tools import TOOL_SCHEMAS, dispatch

SYSTEM_PROMPT = """You are a News & Context Explainer — an AI agent that helps people understand current events by combining breaking news with historical background and relevant numbers.

You have three tools:
- web_search: find the latest news, headlines, and current facts on any topic
- wikipedia_lookup: retrieve historical background, key players, and context for people, places, organisations, or concepts in the news
- calculator: compute statistics, percentages, comparisons, or any numbers mentioned in the news

Today's date is: """ + __import__('datetime').date.today().strftime("%B %d, %Y") + """

Your approach for every question:
1. Search the web for the latest developments on the topic
2. Use Wikipedia to provide historical context and background that explains WHY this matters
3. Use the calculator whenever numbers, percentages, or comparisons will help the user understand the scale or impact
4. Synthesise everything into a clear, structured answer with: current situation, background context, and key numbers

Important: Always cite your sources and indicate the date/recency of information where possible. If search results seem outdated, say so. Be informative but accessible — explain jargon. Your goal is to leave the user genuinely understanding the story, not just knowing the headline."""

MAX_ITERATIONS = 10


def run(user_message: str, history: list[dict] | None = None) -> tuple[str, list[dict]]:
    """
    Run one turn of the ReAct agent loop.

    Args:
        user_message: The latest user query.
        history: Previous conversation messages (list of OpenAI-format dicts).
                 Pass None or [] to start a fresh conversation.

    Returns:
        (final_answer, updated_history)
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    for iteration in range(MAX_ITERATIONS):
        try:
            response = chat_completion(messages, tools=TOOL_SCHEMAS)
        except BadRequestError as e:
            return f"The model produced a malformed response and could not complete the request. Try rephrasing your question.\n\nDetails: {e}", history or []

        # Always append the assistant message (may contain tool_calls)
        messages.append(response)

        tool_calls = extract_tool_calls(response)

        # No tool calls → the model produced its final answer
        if not tool_calls:
            final_answer = response.content or ""
            # Return history without the system prompt so callers can persist it
            conversation_history = messages[1:]
            return final_answer, conversation_history

        # Execute every tool call the model requested
        for call in tool_calls:
            tool_result = dispatch(call["name"], call["arguments"])
            messages.append(
                build_tool_result_message(call["id"], call["name"], tool_result)
            )

    # Safety fallback if MAX_ITERATIONS is hit
    fallback = "I was unable to complete the task within the allowed number of steps."
    return fallback, messages[1:]


def stream(user_message: str, history: list[dict] | None = None):
    """
    Generator version of run() that yields incremental output suitable for a UI.

    Yields dicts with keys:
        {"type": "tool_call",  "name": ..., "arguments": ...}
        {"type": "tool_result","name": ..., "result": ...}
        {"type": "answer",     "content": ..., "history": [...]}
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    for _ in range(MAX_ITERATIONS):
        try:
            response = chat_completion(messages, tools=TOOL_SCHEMAS)
        except BadRequestError as e:
            yield {"type": "answer", "content": f"The model produced a malformed response. Try rephrasing.\n\nDetails: {e}", "history": messages[1:]}
            return
        messages.append(response)

        tool_calls = extract_tool_calls(response)

        if not tool_calls:
            final_answer = response.content or ""
            yield {"type": "answer", "content": final_answer, "history": messages[1:]}
            return

        for call in tool_calls:
            yield {"type": "tool_call", "name": call["name"], "arguments": call["arguments"]}
            tool_result = dispatch(call["name"], call["arguments"])
            yield {"type": "tool_result", "name": call["name"], "result": tool_result}
            messages.append(
                build_tool_result_message(call["id"], call["name"], tool_result)
            )

    yield {
        "type": "answer",
        "content": "I was unable to complete the task within the allowed number of steps.",
        "history": messages[1:],
    }
