"""Build a dev-only RRF run from the already materialized Day 6 dense traces.

This permits reproducible same-question dev comparison without re-embedding.
It is never used for the fresh Day 7 test split; use run_retrieval.py there.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from src.day7 import has_evidence, reciprocal_rank_fusion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--day6-traces", type=Path, default=Path(r"C:\AI_K20\Lab\K5-L3-Day06-2A202603731-DataIngestionEngineering\submission\starter\results\trace_top3.jsonl"))
    parser.add_argument("--bm25-traces", type=Path, required=True)
    parser.add_argument("--questions", type=Path, default=Path(__file__).resolve().parents[1] / "data/questions.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-chars", type=int, default=2400)
    args = parser.parse_args()
    dense_rows = [json.loads(line) for line in args.day6_traces.read_text(encoding="utf-8").splitlines() if line.strip()]
    dense_by_id = {r["question_id"]: r for r in dense_rows if r["strategy"] == "B" and r["split"] == "dev"}
    bm25_rows = json.loads(args.bm25_traces.read_text(encoding="utf-8"))
    bm25_by_id = {r["query_id"]: r for r in bm25_rows if r["strategy"] == "bm25"}
    chunks = json.loads((Path(__file__).resolve().parents[1] / "data/chunks.json").read_text(encoding="utf-8"))
    chunk_by_id = {c["id"]: c for c in chunks}
    questions = [q for q in json.loads(args.questions.read_text(encoding="utf-8")) if q["split"] == "dev" and q["answerable"]]
    traces, dense_traces = [], []
    for q in questions:
        if q["id"] not in dense_by_id or q["id"] not in bm25_by_id:
            raise ValueError(f"Missing dense/BM25 trace for {q['id']}")
        drow, brow = dense_by_id[q["id"]], bm25_by_id[q["id"]]
        if drow["question"] != q["question"] or brow["question"] != q["question"]:
            raise ValueError(f"Question text mismatch for {q['id']}")
        dense = [{"id": h["chunk_id"], "text": h["text"], "source_id": h["metadata"]["document_id"],
                  "source": chunk_by_id[h["chunk_id"]]["source"],
                  "page": str(h["metadata"]["page"]), "score": h["score"], "strategy": "dense_day6"}
                 for h in drow["hits"]]
        for h in dense:
            if h["text"] != chunk_by_id[h["id"]]["text"]:
                raise ValueError(f"Day 6 chunk content mismatch for {h['id']}")
        dense_context, dense_ids, dense_chars = [], [], 0
        for h in dense:
            block = f"[{h['id']} | {h['source']} | page {h['page']}]\n{h['text']}"
            if dense_chars + len(block) <= args.budget_chars:
                dense_context.append(block); dense_ids.append(h["id"]); dense_chars += len(block) + 2
        gold = q["gold"]
        dense_traces.append({"query_id": q["id"], "split": "dev", "strategy": "dense_day6",
                             "question": q["question"], "dense_source": "Day 6 materialized strategy B trace",
                             "results": dense, "context_ids": dense_ids, "context": "\n\n".join(dense_context),
                             "gold": gold, "hit1": bool(dense and has_evidence(dense[0], gold)),
                             "hit3": any(has_evidence(h, gold) for h in dense[:3]),
                             "context_hit": any(has_evidence(h, gold) for h in dense if h["id"] in dense_ids),
                             "context_chars": dense_chars, "elapsed_ms": None})
        fused = reciprocal_rank_fusion(dense, brow["results"], top_k=3)
        used, context_hits, context_chars = set(), [], 0
        for h in fused:
            block = f"[{h['id']} | {h['source']} | page {h['page']}]\n{h['text']}"
            if h["id"] not in used and context_chars + len(block) <= args.budget_chars:
                context_hits.append(h); used.add(h["id"]); context_chars += len(block) + 2
        traces.append({"query_id": q["id"], "split": "dev", "strategy": "hybrid_rrf_cached_day6",
                       "question": q["question"], "dense_source": "Day 6 materialized strategy B trace (same query, corpus, paragraph chunks and Chroma index)",
                       "results": fused, "context_ids": [h["id"] for h in context_hits],
                       "context": "\n\n".join(f"[{h['id']} | {h['source']} | page {h['page']}]\n{h['text']}" for h in context_hits),
                       "gold": gold, "hit1": bool(fused and has_evidence(fused[0], gold)),
                       "hit3": any(has_evidence(h, gold) for h in fused[:3]),
                       "context_hit": any(has_evidence(h, gold) for h in context_hits),
                       "context_chars": context_chars, "elapsed_ms": None})
    n = len(traces)
    summary = {"split": "dev", "method": "RRF over Day 6 dense B top-3 and BM25 top-3",
               "n": n, "hit_at_1": {"n": sum(t["hit1"] for t in traces), "N": n},
               "evidence_hit_at_3": {"n": sum(t["hit3"] for t in traces), "N": n},
               "context_evidence_rate": {"n": sum(t["context_hit"] for t in traces), "N": n},
               "mean_context_chars": statistics.mean([t["context_chars"] for t in traces]) if traces else 0,
               "timing_note": "Dense timings are from Day 6 aggregate run; per-query RRF timing unavailable.",
               "service_cost_usd": 0}
    dense_summary = {"split": "dev", "method": "Day 6 persisted dense collection B, top-3",
                     "n": n, "hit_at_1": {"n": sum(t["hit1"] for t in dense_traces), "N": n},
                     "evidence_hit_at_3": {"n": sum(t["hit3"] for t in dense_traces), "N": n},
                     "context_evidence_rate": {"n": sum(t["context_hit"] for t in dense_traces), "N": n},
                     "mean_context_chars": statistics.mean([t["context_chars"] for t in dense_traces]) if dense_traces else 0,
                     "timing_note": "Existing Day 6 run; per-query timing not stored.", "service_cost_usd": 0}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "dense_day6_traces.json").write_text(json.dumps(dense_traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "dense_day6_summary.json").write_text(json.dumps(dense_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"dense_day6": dense_summary, "hybrid_rrf": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
