import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
from agent.agent import stream

TOOL_ICONS = {
    "calculator": "🧮",
    "wikipedia_lookup": "📖",
    "web_search": "🔍",
}


def format_tool_block(name: str, arguments: dict, result: str) -> str:
    icon = TOOL_ICONS.get(name, "🔧")
    args_str = ", ".join(f"{k}={repr(v)}" for k, v in arguments.items())
    return (
        f"\n<details><summary>{icon} <b>{name}</b>({args_str})</summary>\n\n"
        f"```\n{result.strip()}\n```\n\n</details>\n"
    )


def respond(user_message: str, chat_history: list, agent_history: list):
    """
    Generator that streams agent events into the Gradio chatbot.

    chat_history  : list of {"role": ..., "content": ...} dicts (Gradio 6 messages format)
    agent_history : list of OpenAI-format dicts for the agent loop
    """
    if not user_message.strip():
        yield chat_history, agent_history, ""
        return

    chat_history = chat_history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": ""},
    ]
    yield chat_history, agent_history, ""

    assistant_text = ""
    pending_calls: dict = {}

    for event in stream(user_message, history=agent_history):
        if event["type"] == "tool_call":
            pending_calls[event["name"]] = event["arguments"]

        elif event["type"] == "tool_result":
            name = event["name"]
            arguments = pending_calls.pop(name, {})
            assistant_text += format_tool_block(name, arguments, event["result"])
            chat_history[-1]["content"] = assistant_text
            yield chat_history, agent_history, ""

        elif event["type"] == "answer":
            assistant_text += ("\n" if assistant_text else "") + event["content"]
            chat_history[-1]["content"] = assistant_text
            agent_history = event["history"]
            yield chat_history, agent_history, ""


def clear_conversation():
    return [], [], ""


CSS = """
#chatbot { height: 520px; }
footer { display: none !important; }
"""

THEME = gr.themes.Soft(primary_hue="blue", neutral_hue="slate")

with gr.Blocks(title="News & Context Explainer") as demo:
    agent_history = gr.State([])

    gr.Markdown(
        """
        # 📰 News & Context Explainer
        Paste in a headline or ask about any current event. I'll find the latest news,
        explain the historical background, and put the numbers in perspective.
        Expand the 🔍📖🧮 blocks to see my research steps.
        """
    )

    chatbot = gr.Chatbot(
        elem_id="chatbot",
        label="Conversation",
        height=520,
        render_markdown=True,
        sanitize_html=False,
        buttons=["copy_all"],
        layout="bubble",
    )

    with gr.Row():
        msg_box = gr.Textbox(
            placeholder="Ask me something…",
            show_label=False,
            scale=9,
            autofocus=True,
            lines=1,
        )
        send_btn = gr.Button("Send", variant="primary", scale=1)

    clear_btn = gr.Button("🗑️ Clear conversation", variant="secondary")

    gr.Examples(
        examples=[
            "What's happening with the US-China trade war? Give me the background and what the tariff numbers mean.",
            "Explain the conflict in Ukraine — what's the latest and how did it start?",
            "What is NATO and why is it in the news lately?",
            "What's going on with AI regulation? What are the key laws being discussed and which countries are involved?",
            "Explain the Israel-Gaza conflict — current situation and historical context.",
        ],
        inputs=msg_box,
        label="Try a headline",
    )

    # Wire events
    msg_box.submit(
        respond,
        inputs=[msg_box, chatbot, agent_history],
        outputs=[chatbot, agent_history, msg_box],
    )
    send_btn.click(
        respond,
        inputs=[msg_box, chatbot, agent_history],
        outputs=[chatbot, agent_history, msg_box],
    )
    clear_btn.click(
        clear_conversation,
        outputs=[chatbot, agent_history, msg_box],
    )


if __name__ == "__main__":
    demo.launch(
        share=False,
        show_error=True,
        theme=THEME,
        css=CSS,
    )
