"""
Evaluation harness for the Research Assistant Agent.

Runs all benchmark questions, scores answers, and prints a results table.

Usage:
    python eval/run_eval.py
    python eval/run_eval.py --output eval/results.json
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import re
import time
import argparse
from datetime import datetime
from openai import RateLimitError
from agent.agent import stream
from agent.llm import chat_completion

BENCHMARK_PATH = os.path.join(os.path.dirname(__file__), "benchmark.json")

# ── Scoring ───────────────────────────────────────────────────────────────────

def score_exact(answer: str, expected: str) -> float:
    """Extract a number from the answer and compare to expected value."""
    numbers = re.findall(r"[\d,]+\.?\d*", answer.replace(",", ""))
    for n in numbers:
        try:
            if abs(float(n) - float(expected)) < 0.01:
                return 1.0
        except ValueError:
            continue
    return 0.0


def score_llm_judge(question: str, answer: str, expected: str) -> float:
    """
    Use the LLM itself as a judge.
    Returns 1.0 (correct), 0.5 (partially correct), or 0.0 (incorrect).
    Retries up to 3 times on rate limit errors.
    """
    prompt = f"""You are an impartial grader evaluating an AI assistant's answer.

Question: {question}
Expected answer (key facts): {expected}
Agent's answer: {answer}

Does the agent's answer correctly address the question and contain the key facts?
The agent may answer from its own knowledge OR by using tools — both are acceptable.
Respond with ONLY one of: CORRECT, PARTIAL, INCORRECT
- CORRECT: answer is accurate and contains the key facts
- PARTIAL: answer is on the right track but missing some details or slightly off
- INCORRECT: answer is wrong, irrelevant, or fails to answer the question"""

    messages = [{"role": "user", "content": prompt}]
    for attempt in range(3):
        try:
            response = chat_completion(messages)
            verdict = (response.content or "").strip().upper()
            if "INCORRECT" in verdict:
                return 0.0
            elif "PARTIAL" in verdict:
                return 0.5
            elif "CORRECT" in verdict:
                return 1.0
            return 0.0
        except RateLimitError:
            wait = 30 * (attempt + 1)
            print(f"\n    [rate limit] waiting {wait}s…", end=" ", flush=True)
            time.sleep(wait)
    return 0.0


# ── Runner ────────────────────────────────────────────────────────────────────

def run_single(item: dict) -> dict:
    """Run one benchmark item and return a result dict."""
    question = item["question"]
    tools_called = []
    answer = ""
    error = None
    start = time.time()

    for attempt in range(3):
        try:
            tools_called = []
            for event in stream(question, history=[]):
                if event["type"] == "tool_call":
                    tools_called.append(event["name"])
                elif event["type"] == "answer":
                    answer = event["content"]
            break
        except RateLimitError:
            wait = 30 * (attempt + 1)
            print(f"\n    [rate limit] waiting {wait}s…", end=" ", flush=True)
            time.sleep(wait)
        except Exception as e:
            error = str(e)
            answer = ""
            break

    elapsed = round(time.time() - start, 2)

    if error:
        score = 0.0
    elif item["scoring"] == "exact":
        score = score_exact(answer, item["expected"])
    else:
        score = score_llm_judge(question, answer, item["expected"])

    return {
        "id": item["id"],
        "category": item["category"],
        "question": question,
        "expected": item["expected"],
        "answer": answer[:300] + ("…" if len(answer) > 300 else ""),
        "tools_called": tools_called,
        "num_tool_calls": len(tools_called),
        "score": score,
        "elapsed_s": elapsed,
        "error": error,
    }


def print_results(results: list[dict]):
    total = len(results)
    avg_score = sum(r["score"] for r in results) / total
    avg_tools = sum(r["num_tool_calls"] for r in results) / total
    failures = sum(1 for r in results if r["error"])

    # Per-category breakdown
    categories = {}
    for r in results:
        cat = r["category"]
        categories.setdefault(cat, []).append(r["score"])

    print("\n" + "="*70)
    print("  EVALUATION RESULTS")
    print("="*70)
    print(f"  {'ID':<4} {'Category':<14} {'Score':<8} {'Tools':<6} {'Time':>6}  Question")
    print("-"*70)
    for r in results:
        score_str = f"{r['score']:.1f}"
        tools_str = ",".join(r["tools_called"]) if r["tools_called"] else "none"
        q_short = r["question"][:40] + ("…" if len(r["question"]) > 40 else "")
        flag = "✓" if r["score"] >= 1.0 else ("~" if r["score"] >= 0.5 else "✗")
        print(f"  {r['id']:<4} {r['category']:<14} {flag} {score_str:<6} {r['num_tool_calls']:<6} {r['elapsed_s']:>5}s  {q_short}")

    print("="*70)
    print(f"\n  Overall accuracy:      {avg_score*100:.1f}%  ({sum(r['score'] for r in results):.1f}/{total})")
    print(f"  Avg tool calls/query:  {avg_tools:.1f}")
    print(f"  Failures (errors):     {failures}/{total}")
    print()
    print("  Per-category accuracy:")
    for cat, scores in sorted(categories.items()):
        cat_avg = sum(scores) / len(scores)
        print(f"    {cat:<16}  {cat_avg*100:.1f}%  ({sum(scores):.1f}/{len(scores)})")
    print("="*70 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run agent evaluation benchmark")
    parser.add_argument("--output", help="Save results JSON to this path")
    parser.add_argument("--ids", help="Comma-separated IDs to run (e.g. 1,2,5)")
    parser.add_argument("--from-id", type=int, default=1, help="Resume from this question ID")
    args = parser.parse_args()

    with open(BENCHMARK_PATH) as f:
        benchmark = json.load(f)

    if args.ids:
        ids = {int(i) for i in args.ids.split(",")}
        benchmark = [b for b in benchmark if b["id"] in ids]
    elif args.from_id > 1:
        benchmark = [b for b in benchmark if b["id"] >= args.from_id]

    print(f"\nRunning {len(benchmark)} benchmark questions…")
    print("(This may take a minute — each question hits the LLM + tools)\n")

    results = []
    for i, item in enumerate(benchmark, 1):
        print(f"  [{i}/{len(benchmark)}] Q{item['id']}: {item['question'][:60]}…", end=" ", flush=True)
        result = run_single(item)
        flag = "✓" if result["score"] >= 1.0 else ("~" if result["score"] >= 0.5 else "✗")
        print(f"{flag} (score={result['score']:.1f}, tools={result['num_tool_calls']}, {result['elapsed_s']}s)")
        results.append(result)
        time.sleep(3)  # avoid rate limiting

    print_results(results)

    if args.output:
        output = {
            "run_at": datetime.now().isoformat() + "Z",
            "num_questions": len(results),
            "overall_accuracy": sum(r["score"] for r in results) / len(results),
            "avg_tool_calls": sum(r["num_tool_calls"] for r in results) / len(results),
            "results": results,
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
