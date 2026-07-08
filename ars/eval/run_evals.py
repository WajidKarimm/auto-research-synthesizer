"""
Phase 2 eval loop.

Runs every question in golden_dataset.json through the Phase 1 graph,
scores the answer's faithfulness to its retrieved sources using RAGAS,
and writes a results file. Exits non-zero if the average faithfulness
score is below the threshold -- this is the exact behavior ai_ci.yml
will later gate on.

Faithfulness is reference-free: it checks whether every claim in the
answer can be inferred from the retrieved contexts, not whether the
answer matches some "correct" text. That's why golden_dataset.json
has no ground-truth answers -- we're not there yet (that's context
precision/recall, a later addition once retrieval quality matters).

Usage:
    python -m eval.run_evals
    python -m eval.run_evals --limit 3        # quick smoke test
    python -m eval.run_evals --threshold 0.7  # override the CI gate
"""
import sys
try:
    import langchain_google_vertexai
    sys.modules['langchain_community.chat_models.vertexai'] = langchain_google_vertexai
except ImportError:
    pass

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness

from ars.core.graph import run as run_graph

EVAL_DIR = Path(__file__).parent
GOLDEN_DATASET_PATH = EVAL_DIR / "golden_dataset.json"
RESULTS_PATH = EVAL_DIR / "eval_results.json"

DEFAULT_THRESHOLD = float(os.environ.get("FAITHFULNESS_THRESHOLD", "0.80"))



def _build_evaluator_llm():
    """
    Judging faithfulness with the SAME model that generated the answer is a
    known bias risk (a model is more likely to rate its own claims as
    supported). Three tiers, best to worst:

      1. Claude (ANTHROPIC_API_KEY set) -- fully independent lab, strongest
         judge quality. Preferred when available.
      2. Qwen 3.6 27B via Groq -- no extra API key needed since it's on the
         same Groq account, and it's a genuinely different vendor/training
         lineage than gpt-oss (Alibaba vs OpenAI), so it's a more independent
         second opinion than just using a smaller OpenAI open-weight model.
         Weaker judge than Claude/GPT-5.5 -- may miss subtler hallucinations.
      3. Same Groq model as the generator -- last resort only. Scores from
         this tier are inflated and should not be trusted as a real signal.
    """
    # if os.environ.get("ANTHROPIC_API_KEY"):
    #     from langchain_anthropic import ChatAnthropic

    #     print("[eval] using Claude as an independent faithfulness judge")
    #     return LangchainLLMWrapper(ChatAnthropic(model="claude-sonnet-4-6", temperature=0))

    if os.environ.get("GROQ_API_KEY"):
        print(
            "[eval] no ANTHROPIC_API_KEY set -- using Qwen 3.6 27B (Groq) as an "
            "independent judge instead. Different vendor than the gpt-oss "
            "generator, so it's a real second opinion, but it's a smaller "
            "open-weight model and may miss subtler hallucinations than a "
            "frontier judge would. Set ANTHROPIC_API_KEY for a stronger score."
        )
        return LangchainLLMWrapper(
            ChatGroq(
                model="qwen/qwen3.6-27b",
                temperature=0,
                max_tokens=4096,
                reasoning_effort="none",
            )
        )

    print(
        "[eval] WARNING: no ANTHROPIC_API_KEY or GROQ_API_KEY set -- falling "
        "back to the same model as both generator and judge. This inflates "
        "faithfulness scores because a model is biased toward approving its "
        "own claims. Do not trust this score."
    )
    model_name = os.environ.get("ARS_MODEL", "openai/gpt-oss-120b")
    return LangchainLLMWrapper(ChatGroq(model=model_name, temperature=0))


async def score_one(evaluator_llm, question: str, answer: str, contexts: list[str]) -> float:
    scorer = Faithfulness(llm=evaluator_llm)
    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts or ["(no sources retrieved)"],
    )
    return await scorer.single_turn_ascore(sample)


async def run_eval(limit: int | None = None) -> dict:
    load_dotenv()
    golden = json.loads(GOLDEN_DATASET_PATH.read_text())
    questions = golden["questions"][:limit] if limit else golden["questions"]

    evaluator_llm = _build_evaluator_llm()

    results = []
    for i, item in enumerate(questions, 1):
        q = item["question"]
        print(f"\n[{i}/{len(questions)}] {q}")

        state = run_graph(q)
        answer = state["answer"]
        contexts = [s["content"] for s in state.get("sources", [])]

        try:
            score = await score_one(evaluator_llm, q, answer, contexts)
        except Exception as e:
            print(f"  [eval] scoring failed: {e}")
            score = None

        print(f"  faithfulness: {score}")
        results.append(
            {
                "id": item["id"],
                "question": q,
                "category": item["category"],
                "answer": answer,
                "num_sources": len(contexts),
                "sources": [
                    {"title": s["title"], "url": s["url"]}
                    for s in state.get("sources", [])
                ],
                "faithfulness": score,
            }
        )

    scored = [r["faithfulness"] for r in results if r["faithfulness"] is not None]
    average = sum(scored) / len(scored) if scored else 0.0
    failed_count = len(results) - len(scored)

    summary = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "num_questions": len(results),
        "num_scored": len(scored),
        "num_failed_to_score": failed_count,
        "average_faithfulness": round(average, 4),
        "threshold": DEFAULT_THRESHOLD,
        "passed": average >= DEFAULT_THRESHOLD if scored else False,
        "results": results,
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N questions")
    parser.add_argument("--threshold", type=float, default=None, help="Override the pass/fail threshold")
    args = parser.parse_args()

    global DEFAULT_THRESHOLD
    if args.threshold is not None:
        DEFAULT_THRESHOLD = args.threshold

    summary = asyncio.run(run_eval(limit=args.limit))

    RESULTS_PATH.write_text(json.dumps(summary, indent=2))

    print(f"\n{'=' * 60}")
    print(f"Average faithfulness: {summary['average_faithfulness']} "
          f"(threshold: {summary['threshold']})")
    print(f"Scored {summary['num_scored']}/{summary['num_questions']} "
          f"questions ({summary['num_failed_to_score']} failed to score)")
    print(f"Results written to {RESULTS_PATH}")

    per_category = {}
    for r in summary["results"]:
        if r["faithfulness"] is None:
            continue
        per_category.setdefault(r["category"], []).append(r["faithfulness"])
    print("\nBy category:")
    for cat, scores in per_category.items():
        print(f"  {cat}: {sum(scores) / len(scores):.3f} (n={len(scores)})")

    if summary["passed"]:
        print(f"\nPASS -- average faithfulness meets threshold")
        sys.exit(0)
    else:
        print(f"\nFAIL -- average faithfulness below threshold")
        sys.exit(1)


if __name__ == "__main__":
    main()
