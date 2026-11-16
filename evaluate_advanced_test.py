"""Attach human-review labels and reproducible cost/group metrics to held-out test.

Review labels below were assigned by checking each model answer/citation against
the cited physical PDF page(s), not by an LLM judge or by the model's own output.
This script never changes the benchmark or the locked agent configuration.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADV = ROOT / "data" / "advanced"
RUN = ROOT / "submission" / "results" / "advanced-bm25-test-final"
BENCHMARK = ADV / "benchmark.json"
TRACES = RUN / "traces.json"
REVIEWS = RUN / "reviews.json"
METRICS = RUN / "metrics.json"
RATES = {"cache_hit": 0.003, "cache_miss": 0.15, "output": 0.6}  # USD per 1M tokens

# Human adjudication, with the reason focused on answer completeness, source
# support, and the physical PDF page(s) identified by the citation IDs.
LABELS = {
 "WDR-T01": (True,"30 percent; Table 2.2 row and low-income column are correctly cited on PDF p.84."),
 "WDR-T02": (True,"86 matches the high-income row and e-commerce/e-transactions column on PDF p.227."),
 "WDR-T03": (True,"Correct first recommendation for both safeguard and enabler columns; Table 6.2, PDF p.229."),
 "WDR-T04": (True,"All three Store actions are named correctly and Table 8.1 is cited on PDF p.288."),
 "WDR-T05": (True,"Centralized and decentralized arrangements are both correctly described from Table 8.2, PDF p.292."),
 "WDR-T06": (True,"Purpose and objective of data trusts are accurate and cited to Table 8.3, PDF p.298."),
 "WDR-T07": (True,"The original map note says the solid black rectangles represent cholera deaths; visual inspected on PDF p.47."),
 "WDR-T08": (True,"Map 2.1 caption and adjacent explanation support targeting antipoverty funds; required map page p.77 is cited."),
 "WDR-T09": (True,"Map 2.2 caption and map page support greater bike-path safety; visually inspected on PDF p.79."),
 "WDR-T10": (True,"Figure 1.3 legend correctly maps green to positive and red to negative impacts; PDF p.54."),
 "WDR-T11": (True,"Figure 1.4 explicitly gives national and international levels; PDF p.59."),
 "WDR-T12": (True,"Figure 1.1 supports the 1-in-10 to 9-in-10 trend; the added recent-decade percentages are visible in its chart on PDF p.46."),
 "WDR-T13": (True,"Figure 2.4 caption answers the comparison and the cited PDF p.82 provides the numeric context."),
 "WDR-T14": (True,"The figure caption and plotted values support the finding that most plans are not fully funded; PDF p.86."),
 "WDR-T15": (True,"Figure 2.9 caption and plotted correlations support the stated positive relationships; PDF pp.89-90."),
 "WDR-T16": (True,"Figure 7.4 supports exponential growth and about half of services trade; PDF p.258."),
 "WDR-T17": (True,"Combines a positive pathway from Figure 1.3 and the two policy levels from Figure 1.4; both required pages are cited."),
 "WDR-T18": (True,"The Croatia map passage supports targeted EU funds; Table 2.2 supports 30% advance-calendar and 63% metadata values; pp.76-77,84 cited."),
 "WDR-T19": (False,"Abstained despite answerable evidence: Figure 7.4 on PDF p.258 and Figure 7.6 on p.268 support both parts."),
 "WDR-T20": (True,"Correct preservation action and centralized-approach example are supported by Tables 8.1/8.2; PDF pp.288,292 cited."),
 "WDR-T21": (True,"All requested administrative-data examples are correctly summarized from Box 2.1; PDF p.75 cited."),
 "WDR-T22": (True,"Representativeness and inferential-analytics format are both stated in the cited discussion; PDF pp.47-48."),
 "WDR-T23": (True,"All listed safeguards and purposes match the paragraph; primary evidence page PDF p.79 is cited."),
 "WDR-T24": (True,"Alignment and implementation steps are accurately stated from Figure 8.2 discussion; PDF p.290 cited."),
 "WDR-T25": (True,"Correct abstention; exact 2025 DeepSeek market-share claim is absent from the 2021 report."),
 "WDR-T26": (True,"Correct abstention; occupation-specific Vietnam salary claim is not reported."),
 "WDR-T27": (True,"Correct abstention; exact Facebook account count for Vietnam is not reported."),
 "WDR-T28": (True,"Correct abstention; the report provides no exact global 2020 ransomware-payment total."),
 "WDR-T29": (True,"Correct abstention; Vietnam government data-scientist employment count is not reported."),
 "WDR-T30": (True,"Correct abstention; the report predates DeepSeek and has no 2025 revenue evidence."),
}


def cost_for(trace: dict) -> tuple[float, dict]:
    usage_sum = Counter()
    for call in trace.get("calls", []):
        usage = call.get("usage") or {}
        detail = usage.get("prompt_tokens_details") or {}
        prompt = int(usage.get("prompt_tokens", 0) or 0)
        hit = int(usage.get("prompt_cache_hit_tokens", detail.get("cached_tokens", 0)) or 0)
        miss = int(usage.get("prompt_cache_miss_tokens", max(0, prompt - hit)) or 0)
        usage_sum["requests"] += 1
        usage_sum["prompt_tokens"] += prompt
        usage_sum["cache_hit_tokens"] += hit
        usage_sum["cache_miss_tokens"] += miss
        usage_sum["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
        usage_sum["total_tokens"] += int(usage.get("total_tokens", 0) or 0)
    usd = (usage_sum["cache_hit_tokens"] * RATES["cache_hit"]
           + usage_sum["cache_miss_tokens"] * RATES["cache_miss"]
           + usage_sum["completion_tokens"] * RATES["output"]) / 1_000_000
    return round(usd, 8), dict(usage_sum)


def main() -> None:
    benchmark = {q["id"]: q for q in json.loads(BENCHMARK.read_text(encoding="utf-8")) if q["split"] == "test"}
    traces = json.loads(TRACES.read_text(encoding="utf-8"))
    chunks = {c["id"]: c for c in json.loads((ADV / "chunks.json").read_text(encoding="utf-8"))}
    reviews = []
    groups: dict[str, Counter] = {}
    total = Counter()
    total_usage = Counter()
    total_elapsed = 0.0
    for trace in traces:
        q = benchmark[trace["query_id"]]
        expected_pages = {str(g["page"]) for g in q.get("gold", [])}
        cited_pages = {str(chunks[cid]["page"]) for cid in trace.get("citations", []) if cid in chunks}
        label, reason = LABELS[trace["query_id"]]
        cost, usage = cost_for(trace)
        trace["cost_usd"] = cost
        trace["cost_basis"] = "DeepSeek Flash off-peak rates, USD per 1M tokens"
        trace["review"] = {"pass": label, "reason": reason,
                           "gold_pages": sorted(expected_pages, key=int),
                           "cited_pages": sorted(cited_pages, key=int)}
        if q["answerable"] and label and not expected_pages.issubset(cited_pages):
            raise ValueError(f"{q['id']} lacks citations to required gold pages")
        if not q["answerable"] and trace.get("status") != "abstain":
            raise ValueError(f"Unanswerable case {q['id']} did not abstain")
        if trace.get("status") == "error":
            label = False
        group = q["primary_group"]
        stats = groups.setdefault(group, Counter())
        stats["total"] += 1
        stats["pass"] += int(label)
        stats["failed"] += int(not label)
        stats["elapsed_ms"] += float(trace.get("elapsed_ms", 0) or 0)
        stats["cost_usd"] += cost
        total["total"] += 1
        total["pass"] += int(label)
        total["failed"] += int(not label)
        total["errors"] += int(trace.get("status") == "error")
        total["answerable"] += int(q["answerable"])
        total["answerable_pass"] += int(q["answerable"] and label)
        total["unanswerable"] += int(not q["answerable"])
        total["unanswerable_pass"] += int(not q["answerable"] and label)
        total_elapsed += float(trace.get("elapsed_ms", 0) or 0)
        total_usage.update(usage)
        reviews.append({"id": q["id"], "primary_group": group, "answerable": q["answerable"],
                        "pass": label, "reason": reason, "gold_pages": sorted(expected_pages, key=int),
                        "cited_pages": sorted(cited_pages, key=int), "status": trace.get("status"),
                        "elapsed_ms": trace.get("elapsed_ms"), "cost_usd": cost,
                        "answer": trace.get("answer"), "error": trace.get("error")})
    TRACES.write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REVIEWS.write_text(json.dumps(reviews, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    denominator = int(total["total"])
    metrics = {
        "model": "deepseek-flash", "retrieval": "bm25_locator_aware", "test_split_locked_before_run": True,
        "total_test": {"pass": int(total["pass"]), "N": denominator,
                       "accuracy_percent": round(100 * total["pass"] / denominator, 4)},
        "answerable": {"pass": int(total["answerable_pass"]), "N": int(total["answerable"])},
        "unanswerable_abstention": {"pass": int(total["unanswerable_pass"]), "N": int(total["unanswerable"])},
        "system_errors": int(total["errors"]), "groups": {
            g: {"pass": int(v["pass"]), "N": int(v["total"]), "accuracy_percent": round(100*v["pass"]/v["total"],4)}
            for g,v in groups.items()},
        "elapsed_ms_total": round(total_elapsed, 2), "elapsed_seconds_total": round(total_elapsed/1000,2),
        "estimated_cost_usd": round(sum(v["cost_usd"] for v in groups.values()), 8),
        "usage": dict(total_usage), "rates_usd_per_1m_tokens": RATES,
        "price_note": "Estimate from per-request API usage. The API provider's live billed amount may differ due to pricing/time changes.",
        "manual_review_file": REVIEWS.name, "trace_file": TRACES.name,
        "failure_ids": [r["id"] for r in reviews if not r["pass"]],
    }
    METRICS.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
