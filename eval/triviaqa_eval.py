"""
Agent vs. No-Agent evaluation on TriviaQA (100 questions).

Scoring: a prediction is CORRECT if any accepted answer string appears
in the normalised prediction (case-insensitive, punctuation-stripped).
This is the standard TriviaQA "answer-in-prediction" metric.

Usage:
    python eval/triviaqa_eval.py
    python eval/triviaqa_eval.py --output eval/triviaqa_results.json
    python eval/triviaqa_eval.py --n 25          # run only 25 questions
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import json
import time
import argparse
from datetime import datetime
from openai import RateLimitError
from datasets import load_dataset
from agent.agent import stream
from agent.llm import chat_completion

# ── Scoring ───────────────────────────────────────────────────────────────────

def normalise(text: str) -> str:
    """Lowercase, strip punctuation and extra whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_correct(prediction: str, accepted_answers: list[str]) -> bool:
    """Return True if any accepted answer appears in the normalised prediction."""
    norm_pred = normalise(prediction)
    return any(normalise(ans) in norm_pred for ans in accepted_answers)


# ── LLM calls with retry ──────────────────────────────────────────────────────

def run_agent(question: str) -> tuple[str, list[str]]:
    """Run the full agent (with tools). Returns (answer, tools_called)."""
    answer = ""
    tools_called = []
    for attempt in range(3):
        try:
            tools_called = []
            for event in stream(question, history=[]):
                if event["type"] == "tool_call":
                    tools_called.append(event["name"])
                elif event["type"] == "answer":
                    answer = event["content"]
            return answer, tools_called
        except RateLimitError:
            wait = 30 * (attempt + 1)
            print(f"\n    [rate limit] waiting {wait}s…", end=" ", flush=True)
            time.sleep(wait)
    return "", []


def run_no_agent(question: str) -> str:
    """Run the bare LLM with no tools. Returns the answer string."""
    messages = [
        {"role": "system", "content": "Answer the following question concisely and accurately."},
        {"role": "user", "content": question},
    ]
    for attempt in range(3):
        try:
            response = chat_completion(messages, tools=None)
            return response.content or ""
        except RateLimitError:
            wait = 30 * (attempt + 1)
            print(f"\n    [rate limit] waiting {wait}s…", end=" ", flush=True)
            time.sleep(wait)
    return ""


# ── Results table ─────────────────────────────────────────────────────────────

def print_results(results: list[dict]):
    n = len(results)
    agent_correct = sum(1 for r in results if r["agent_correct"])
    no_agent_correct = sum(1 for r in results if r["no_agent_correct"])
    agent_acc = agent_correct / n * 100
    no_agent_acc = no_agent_correct / n * 100
    improvement = agent_acc - no_agent_acc
    avg_tools = sum(r["num_tool_calls"] for r in results) / n

    print("\n" + "=" * 70)
    print("  TRIVIAQA EVALUATION — Agent vs. No-Agent  (n={})".format(n))
    print("=" * 70)
    print(f"  {'Metric':<35} {'No-Agent':>10} {'Agent':>10}")
    print("-" * 70)
    print(f"  {'Accuracy':<35} {no_agent_acc:>9.1f}% {agent_acc:>9.1f}%")
    print(f"  {'Correct answers':<35} {no_agent_correct:>10} {agent_correct:>10}")
    print(f"  {'Avg tool calls per question':<35} {'—':>10} {avg_tools:>9.1f}")
    print("-" * 70)
    delta_str = f"+{improvement:.1f}%" if improvement >= 0 else f"{improvement:.1f}%"
    print(f"  Agent improvement over no-agent: {delta_str}")
    print("=" * 70 + "\n")

    # Per-question breakdown (first 20 only to keep output readable)
    print("  Sample results (first 20 questions):")
    print(f"  {'ID':<5} {'Agent':^7} {'LLM':^7} {'Tools':<20} Question")
    print("  " + "-" * 65)
    for r in results[:20]:
        a_flag = "✓" if r["agent_correct"] else "✗"
        l_flag = "✓" if r["no_agent_correct"] else "✗"
        tools = ",".join(r["tools_called"]) if r["tools_called"] else "none"
        q = r["question"][:35] + ("…" if len(r["question"]) > 35 else "")
        print(f"  {r['id']:<5} {a_flag:^7} {l_flag:^7} {tools:<20} {q}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100, help="Number of questions to evaluate")
    parser.add_argument("--output", default="eval/triviaqa_results.json")
    args = parser.parse_args()

    print(f"\nLoading TriviaQA validation set…")
    ds = load_dataset("trivia_qa", "rc", split=f"validation[:{args.n}]", trust_remote_code=True)
    print(f"Loaded {len(ds)} questions.\n")
    print(f"Running agent vs. no-agent on {len(ds)} questions (~40 min for 100)…\n")

    results = []
    for i, item in enumerate(ds, 1):
        question = item["question"]
        accepted = item["answer"]["aliases"]  # list of valid answer strings

        print(f"  [{i:>3}/{len(ds)}] {question[:60]}…", end=" ", flush=True)

        agent_answer, tools_called = run_agent(question)
        time.sleep(3)
        no_agent_answer = run_no_agent(question)
        time.sleep(3)

        a_correct = is_correct(agent_answer, accepted)
        l_correct = is_correct(no_agent_answer, accepted)

        a_flag = "✓" if a_correct else "✗"
        l_flag = "✓" if l_correct else "✗"
        print(f"agent={a_flag} llm={l_flag} tools={len(tools_called)}")

        results.append({
            "id": i,
            "question": question,
            "accepted_answers": accepted[:5],
            "agent_answer": agent_answer[:300],
            "no_agent_answer": no_agent_answer[:300],
            "tools_called": tools_called,
            "num_tool_calls": len(tools_called),
            "agent_correct": a_correct,
            "no_agent_correct": l_correct,
        })

    print_results(results)

    output = {
        "run_at": datetime.now().isoformat(),
        "dataset": "TriviaQA (rc, validation)",
        "n": len(results),
        "agent_accuracy": sum(r["agent_correct"] for r in results) / len(results),
        "no_agent_accuracy": sum(r["no_agent_correct"] for r in results) / len(results),
        "avg_tool_calls": sum(r["num_tool_calls"] for r in results) / len(results),
        "results": results,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
