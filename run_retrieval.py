"""Run Day 7 dense, BM25 and hybrid retrieval against one benchmark split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.day7 import Day6DenseIndex, run_retrieval


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--day6-root", type=Path, default=Path(r"C:\AI_K20\Lab\K5-L3-Day06-2A202603731-DataIngestionEngineering"))
    parser.add_argument("--chunks", type=Path, default=Path(__file__).resolve().parents[1] / "data/chunks.json")
    parser.add_argument("--questions", type=Path, default=Path(__file__).resolve().parents[1] / "data/questions.json")
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--budget-chars", type=int, default=2400)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    questions = [q for q in json.loads(args.questions.read_text(encoding="utf-8"))
                 if q["split"] == args.split and q["answerable"]]
    dense = Day6DenseIndex(args.day6_root)
    traces, _ = run_retrieval(chunks, questions, dense, top_k=args.top_k, budget_chars=args.budget_chars)
    summaries = {}
    for method in ("dense_day6", "bm25", "hybrid_rrf"):
        rows = [t for t in traces if t["strategy"] == method]
        n = len(rows)
        summaries[method] = {
            "split": args.split, "n": n,
            "hit_at_1": {"n": sum(t["hit1"] for t in rows), "N": n},
            "evidence_hit_at_3": {"n": sum(t["hit3"] for t in rows), "N": n},
            "context_evidence_rate": {"n": sum(bool(t["context_hit"]) for t in rows), "N": n},
            "mean_context_chars": sum(t["context_chars"] for t in rows) / n if n else 0,
            "mean_elapsed_ms": sum(t["elapsed_ms"] for t in rows) / n if n else 0,
            "service_cost_usd": 0,
        }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "summary.json").write_text(json.dumps({"split": args.split, "top_k": args.top_k,
       "budget_chars": args.budget_chars, "methods": summaries}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
