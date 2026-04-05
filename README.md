# News & Context Explainer — LLM Agent from Scratch

A **News & Context Explainer** agent built from scratch — no LangChain, no CrewAI. Just a custom ReAct loop, direct LLM API calls, and plain Python.

---

## What It Does

People read headlines but rarely understand the full story. This agent bridges that gap: given any current events question, it autonomously searches for the latest news, retrieves historical background, and puts key numbers in perspective — synthesising everything into one clear, cited answer.

**Example:** *"What's happening with the US-China trade war and what do the tariff numbers actually mean?"*
1. Searches the web for the latest tariff headlines
2. Looks up the history of US-China trade relations on Wikipedia
3. Calculates the real dollar impact of the tariffs
4. Returns a structured answer: current situation + background context + numbers

---

## Architecture

### Agent Loop (ReAct-style)

A `Reason + Act` loop written entirely from scratch:

1. **Think** — LLM receives system prompt + conversation history + tool definitions, outputs a reasoning step and a tool call (or final answer)
2. **Act** — execute the tool call, append result to context
3. **Observe** — feed tool result back to LLM
4. Repeat until the LLM emits a final answer

### Tools

| Tool | Role |
|---|---|
| `web_search` | Find **current news** — real-time results via Tavily Search API |
| `wikipedia_lookup` | Retrieve **historical background** — key players, origins, context |
| `calculator` | Crunch **numbers in the news** — tariff costs, GDP comparisons, percentages |

Each tool is a plain Python function with a JSON schema description passed to the LLM.

### User Interface

A clean web UI built with **Gradio**:
- Chat input with example news prompts
- Expandable tool call trace (🔍📖🧮) so users can see exactly how the agent researched the answer
- Clear conversation button for a fresh session

### Evaluation

An eval harness with 20 benchmark questions across 3 categories (news-themed).

**Metrics:**
- **Answer accuracy** — exact match for math; LLM-as-judge for open-ended
- **Tool call efficiency** — average number of tool calls per query
- **Failure rate** — % of queries that error or crash

---

## File Structure

```
LLM_agent_from_scratch/
├── agent/
│   ├── agent.py          # Core ReAct loop (run + stream)
│   ├── llm.py            # LLM API wrapper (Groq via OpenAI-compatible API)
│   └── tools/
│       ├── __init__.py   # Tool registry + dispatch
│       ├── search.py     # Tavily real-time web search
│       ├── calculator.py # Safe AST-based math evaluator
│       └── wikipedia.py  # Wikipedia article fetcher
├── ui/
│   └── app.py            # Gradio 6 chat UI
├── eval/
│   ├── benchmark.json        # 20 news-themed questions + expected answers
│   ├── run_eval.py           # Evaluation harness
│   ├── triviaqa_eval.py      # Agent vs no-agent on TriviaQA (100 questions)
│   ├── results.json          # News benchmark results
│   └── triviaqa_results.json # TriviaQA eval results
├── presentation/
│   ├── slides.tex            # Beamer LaTeX slides
│   ├── slides.pdf            # Compiled presentation
│   └── plot_eval.py          # Generates eval bar chart (matplotlib)
├── .env.example          # API key template
├── requirements.txt
└── README.md
```

---

## Design Decisions

- **LLM**: `qwen/qwen3-32b` via Groq's free OpenAI-compatible API — fast, strong tool-calling, no cost
- **Web search**: Tavily Search API — purpose-built for LLM agents, real-time results with published dates
- **Tool format**: OpenAI JSON Schema function-calling — works with any OpenAI-compatible endpoint
- **Agent loop**: Custom ReAct (Reason + Act), max 10 iterations, with `BadRequestError` guard
- **UI**: Gradio 6 — minimal setup, clean bubble layout, collapsible tool traces
- **Eval scoring**: Agent vs. no-agent comparison on TriviaQA (100 questions), answer-in-prediction metric

---

## How to Run

```bash
# Install dependencies
python3 -m pip install -r requirements.txt --break-system-packages

# Copy the env template and fill in your keys
cp .env.example .env
# edit .env and set:
#   GROQ_API_KEY=your_groq_key
#   TAVILY_API_KEY=your_tavily_key

# Launch the UI
python3 ui/app.py
# Open http://127.0.0.1:7860 in your browser

# Run evaluation (agent vs no-agent on TriviaQA)
python3 eval/triviaqa_eval.py --n 100 --output eval/triviaqa_results.json
```

---

## Evaluation Results

### Agent vs. No-Agent — TriviaQA (100 questions)

Evaluation on the public [TriviaQA](https://huggingface.co/datasets/trivia_qa) dataset (validation set). Each question was run twice — once with the agent (tools enabled) and once with a plain LLM call (no tools). Scoring uses the standard TriviaQA "answer-in-prediction" metric: a response is correct if any accepted answer string appears in the normalised prediction.

| Metric | No-Agent (plain LLM) | Agent (with tools) |
|---|---|---|
| **Accuracy** | 77% | **83%** |
| Correct answers | 77 / 100 | 83 / 100 |
| Avg tool calls | — | 1.0 |
| Crashes / errors | 0 | 0 |

**Agent improvement: +6% over bare LLM**

The agent correctly answered 6 additional questions that the plain LLM got wrong by using Wikipedia or web search to look up the answer. Full results are in `eval/triviaqa_results.json`.
