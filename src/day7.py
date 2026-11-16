"""Reusable Day 7 retrieval experiments over the Day 6 Sherlock corpus."""

from __future__ import annotations

import json
import math
import re
import statistics
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


def tokens(text: str) -> list[str]:
    return re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold())


def has_evidence(chunk: dict[str, Any], gold: list[dict[str, Any]]) -> bool:
    normalize = lambda value: " ".join(str(value).split())
    return any(
        str(chunk["source_id"]) == str(g["source_id"])
        and str(chunk["page"]) == str(g["page"])
        and normalize(g["evidence"]) in normalize(chunk["text"])
        for g in gold
    )


class LexicalIndex:
    """BM25-style lexical retrieval using only the Python standard library."""

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks = chunks
        self.rows = [Counter(tokens(c["text"])) for c in chunks]
        self.lengths = [sum(row.values()) for row in self.rows]
        self.avg_len = statistics.mean(self.lengths) if self.lengths else 0.0
        self.df = Counter(term for row in self.rows for term in row)

    def search(self, query: str, top_k: int = 3, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        qterms = set(tokens(query))
        scored = []
        n = len(self.chunks)
        for c, row, length in zip(self.chunks, self.rows, self.lengths):
            if filters and any(str(c.get(k)) != str(v) for k, v in filters.items()):
                continue
            score = 0.0
            for term in qterms:
                freq = row.get(term, 0)
                if not freq:
                    continue
                idf = math.log(1 + (n - self.df[term] + 0.5) / (self.df[term] + 0.5))
                denom = freq + 1.2 * (1 - 0.75 + 0.75 * length / (self.avg_len or 1))
                score += idf * freq * 2.2 / denom
            if score:
                scored.append({**c, "score": score, "strategy": "bm25"})
        return sorted(scored, key=lambda c: (-c["score"], c["id"]))[:top_k]


def reciprocal_rank_fusion(*rankings: list[dict[str, Any]], top_k: int = 3, k: int = 60) -> list[dict[str, Any]]:
    """Fuse ranked results without assuming lexical/dense scores share a scale."""
    by_id: dict[str, dict[str, Any]] = {}
    scores: Counter[str] = Counter()
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            cid = hit["id"]
            scores[cid] += 1.0 / (k + rank)
            by_id.setdefault(cid, hit)
    return [{**by_id[cid], "score": scores[cid], "strategy": "hybrid_rrf"}
            for cid in sorted(scores, key=lambda cid: (-scores[cid], cid))[:top_k]]


class Day6DenseIndex:
    """Query the existing Day 6 Chroma collection using its original embedder."""

    def __init__(self, day6_root: Path, collection: str = "day6_sherlock_b") -> None:
        import importlib.util
        day6_root = day6_root.resolve()
        import chromadb

        manifest = json.loads((day6_root / "starter/data/manifest.json").read_text(encoding="utf-8"))
        model = json.loads((day6_root / "submission/starter/results/experiment_report.json").read_text(encoding="utf-8"))["embedding"]["model"]
        embedding_path = day6_root / "submission/starter/src/embeddings.py"
        spec = importlib.util.spec_from_file_location("day6_embedding_adapter", embedding_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load Day 6 embedder from {embedding_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.embedder = module.MyEmbedder(model)
        self.collection = chromadb.PersistentClient(
            path=str(day6_root / "submission/starter/results/chroma_controlled")
        ).get_collection(collection, embedding_function=None)
        self.manifest = {s["source_id"]: s["title"] for s in manifest["sources"]}

    def search(self, query: str, top_k: int = 3, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        where = None
        if filters and "source_id" in filters:
            where = {"document_id": str(filters["source_id"])}
        result = self.collection.query(
            query_embeddings=[self.embedder.embed_query(query)], n_results=top_k,
            where=where, include=["documents", "metadatas", "distances"],
        )
        rows = []
        for cid, text, meta, distance in zip(result["ids"][0], result["documents"][0],
                                               result["metadatas"][0], result["distances"][0]):
            source_id = str(meta.get("document_id", ""))
            rows.append({"id": cid, "text": text, "source_id": source_id,
                         "source": self.manifest.get(source_id, source_id),
                         "page": str(meta.get("page", "")), "metadata": meta,
                         "score": 1.0 - float(distance), "strategy": "dense_day6"})
        return rows


def run_retrieval(chunks: list[dict[str, Any]], questions: list[dict[str, Any]],
                  dense: Day6DenseIndex, *, top_k: int = 3, budget_chars: int = 2400) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    lexical = LexicalIndex(chunks)
    traces = []
    for q in questions:
        question, filters = q["question"], q.get("filter")
        start = time.perf_counter()
        dense_hits = dense.search(question, max(10, top_k), filters)
        dense_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        lexical_hits = lexical.search(question, max(10, top_k), filters)
        lexical_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        methods = {
            "dense_day6": dense_hits[:top_k],
            "bm25": lexical_hits[:top_k],
            "hybrid_rrf": reciprocal_rank_fusion(dense_hits, lexical_hits, top_k=top_k),
        }
        for name, hits in methods.items():
            method_ms = dense_ms if name == "dense_day6" else lexical_ms if name == "bm25" else 0.0
            if name == "hybrid_rrf":
                method_ms = (time.perf_counter() - start) * 1000
            gold = q.get("gold", [])
            context_hits, used, length = [], set(), 0
            for h in hits:
                block_len = len(h["text"]) + len(h["source"]) + len(h["page"]) + 20
                if h["id"] not in used and length + block_len <= budget_chars:
                    context_hits.append(h)
                    used.add(h["id"])
                    length += block_len
            traces.append({"query_id": q["id"], "split": q["split"], "strategy": name,
                           "question": question, "answerable": q.get("answerable", True),
                           "gold": gold, "results": hits, "context_ids": [h["id"] for h in context_hits],
                           "context_hit": any(has_evidence(h, gold) for h in context_hits) if gold else None,
                           "hit1": bool(hits and gold and has_evidence(hits[0], gold)),
                           "hit3": any(has_evidence(h, gold) for h in hits[:3]) if gold else None,
                           "context_chars": length, "filter": filters,
                           "elapsed_ms": method_ms})
    summaries = {}
    for name in ("dense_day6", "bm25", "hybrid_rrf"):
        rows = [t for t in traces if t["strategy"] == name and t["split"] == "dev" and t["answerable"]]
        n = len(rows)
        summaries[name] = {"split": "dev", "n": n,
                           "hit_at_1": {"n": sum(t["hit1"] for t in rows), "N": n},
                           "evidence_hit_at_3": {"n": sum(t["hit3"] for t in rows), "N": n},
                           "context_evidence_rate": {"n": sum(t["context_hit"] for t in rows), "N": n},
                           "mean_context_chars": statistics.mean([t["context_chars"] for t in rows]) if rows else 0,
                           "median_elapsed_ms": statistics.median([t["elapsed_ms"] for t in rows]) if rows else 0,
                           "service_cost_usd": 0}
    return traces, summaries

