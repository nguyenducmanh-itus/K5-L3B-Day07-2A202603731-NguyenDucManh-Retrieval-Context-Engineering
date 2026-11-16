"""Offline lexical baselines for use before the model/vector dependencies are ready."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.day7 import LexicalIndex, has_evidence
from starter.baseline import TfidfIndex, pack_context


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=Path, default=Path(__file__).resolve().parents[1] / "data/chunks.json")
    parser.add_argument("--questions", type=Path, default=Path(__file__).resolve().parents[1] / "data/questions.json")
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--budget-chars", type=int, default=2400)
    args = parser.parse_args()
    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    questions = [q for q in json.loads(args.questions.read_text(encoding="utf-8"))
                 if q["split"] == args.split and q["answerable"]]
    indexes = {"tfidf": TfidfIndex(chunks), "bm25": LexicalIndex(chunks)}
    traces = []
    for q in questions:
        for method, index in indexes.items():
            start = time.perf_counter()
            if method == "tfidf":
                hits = index.search(q["question"], args.top_k)
                # Normalize starter result records to the Day 7 trace schema.
                hits = [{**h, "id": h["id"] if "id" in h else h["chunk_id"]} for h in hits]
            else:
                hits = index.search(q["question"], args.top_k, q.get("filter"))
            elapsed_ms = (time.perf_counter() - start) * 1000
            context, context_ids = pack_context(hits, args.budget_chars)
            gold = q["gold"]
            traces.append({"query_id": q["id"], "split": args.split, "strategy": method,
                           "question": q["question"], "results": hits, "context": context,
                           "context_ids": context_ids, "gold": gold,
                           "hit1": bool(hits and has_evidence(hits[0], gold)),
                           "hit3": any(has_evidence(h, gold) for h in hits[:3]),
                           "context_hit": any(has_evidence(h, gold) and h["id"] in context_ids for h in hits),
                           "context_chars": len(context), "elapsed_ms": elapsed_ms})
    summaries = {}
    for method in indexes:
        rows = [t for t in traces if t["strategy"] == method]
        n = len(rows)
        summaries[method] = {"split": args.split, "n": n,
                             "hit_at_1": {"n": sum(t["hit1"] for t in rows), "N": n},
                             "evidence_hit_at_3": {"n": sum(t["hit3"] for t in rows), "N": n},
                             "context_evidence_rate": {"n": sum(t["context_hit"] for t in rows), "N": n},
                             "mean_context_chars": statistics.mean([t["context_chars"] for t in rows]) if rows else 0,
                             "median_elapsed_ms": statistics.median([t["elapsed_ms"] for t in rows]) if rows else 0,
                             "service_cost_usd": 0}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
