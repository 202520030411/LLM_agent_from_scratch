import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Groq is accessed via its OpenAI-compatible REST endpoint.
# This means we can keep all our tool schemas in OpenAI format unchanged.
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "qwen/qwen3-32b"

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add your Groq API key."
            )
        _client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
    return _client


def chat_completion(messages: list[dict], tools: list[dict] | None = None, model: str = DEFAULT_MODEL) -> dict:
    """
    Send a chat completion request to Gemini (via OpenAI-compatible API).
    Returns the raw message object from the API response.
    """
    kwargs = {
        "model": model,
        "messages": messages,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    response = _get_client().chat.completions.create(**kwargs)
    return response.choices[0].message


def extract_tool_calls(message) -> list[dict] | None:
    """
    Extract tool call instructions from an assistant message.
    Returns a list of dicts with 'id', 'name', and 'arguments' (parsed JSON), or None.
    """
    if not message.tool_calls:
        return None

    calls = []
    for tc in message.tool_calls:
        calls.append({
            "id": tc.id,
            "name": tc.function.name,
            "arguments": json.loads(tc.function.arguments),
        })
    return calls


def build_tool_result_message(tool_call_id: str, name: str, result: str) -> dict:
    """Build the 'tool' role message that feeds a result back to the model."""
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": name,
        "content": result,
    }
