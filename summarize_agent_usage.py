"""Summarize DeepSeek request usage and an off-peak cost estimate."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ["agent-bm25-dev", "agent-bm25-dev-audited", "agent-bm25-dev-final",
        "agent-tfidf-dev", "agent-tfidf-test", "agent-bm25-test"]
RATES_PER_MILLION = {"cache_hit": 0.003, "cache_miss": 0.15, "output": 0.6}


def summarize(run_dir: Path) -> dict:
    traces = json.loads((run_dir / "traces.json").read_text(encoding="utf-8"))
    result = {"requests": 0, "prompt_tokens": 0, "cache_hit_tokens": 0,
              "cache_miss_tokens": 0, "completion_tokens": 0,
              "total_tokens": 0, "elapsed_ms": sum(t.get("elapsed_ms", 0) for t in traces)}
    for trace in traces:
        for call in trace.get("calls", []):
            usage = call.get("usage") or {}
            detail = usage.get("prompt_tokens_details") or {}
            prompt = int(usage.get("prompt_tokens", 0))
            hit = int(usage.get("prompt_cache_hit_tokens", detail.get("cached_tokens", 0)) or 0)
            miss = int(usage.get("prompt_cache_miss_tokens", max(0, prompt - hit)) or 0)
            result["requests"] += 1
            result["prompt_tokens"] += prompt
            result["cache_hit_tokens"] += hit
            result["cache_miss_tokens"] += miss
            result["completion_tokens"] += int(usage.get("completion_tokens", 0))
            result["total_tokens"] += int(usage.get("total_tokens", 0))
    result["elapsed_seconds"] = round(result["elapsed_ms"] / 1000, 2)
    result["estimated_cost_usd"] = round(
        result["cache_hit_tokens"] * RATES_PER_MILLION["cache_hit"] / 1_000_000
        + result["cache_miss_tokens"] * RATES_PER_MILLION["cache_miss"] / 1_000_000
        + result["completion_tokens"] * RATES_PER_MILLION["output"] / 1_000_000,
        7,
    )
    return result


def main() -> None:
    runs = {name: summarize(ROOT / "submission/results" / name) for name in RUNS}
    total = {k: sum(r[k] for r in runs.values()) for k in
             ("requests", "prompt_tokens", "cache_hit_tokens", "cache_miss_tokens", "completion_tokens", "total_tokens", "elapsed_ms")}
    total["elapsed_seconds"] = round(total["elapsed_ms"] / 1000, 2)
    total["estimated_cost_usd"] = round(sum(r["estimated_cost_usd"] for r in runs.values()), 7)
    report = {"model": "deepseek-flash", "price_basis": "DeepSeek listed off-peak rates on 2026-11-16",
              "rates_usd_per_1m_tokens": RATES_PER_MILLION, "runs": runs, "total_including_dev_reruns": total,
              "final_evaluation_runs": {name: runs[name] for name in ("agent-tfidf-dev", "agent-bm25-dev-final", "agent-tfidf-test", "agent-bm25-test")},
              "final_evaluation_cost_usd": round(sum(runs[name]["estimated_cost_usd"] for name in
                  ("agent-tfidf-dev", "agent-bm25-dev-final", "agent-tfidf-test", "agent-bm25-test")), 7)}
    output = ROOT / "submission/results/usage_summary.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"all_runs": total, "final_evaluation_cost_usd": report["final_evaluation_cost_usd"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
