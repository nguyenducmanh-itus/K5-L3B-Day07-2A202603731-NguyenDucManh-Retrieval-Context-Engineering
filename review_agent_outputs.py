"""Create a transparent first-pass review for the captured agent runs.

Scores below are assistant-prepared annotations and should be checked by the
student before submission. Citation support is reviewed separately from answer
correctness; D7T002's correct species answer is not supported by its cited
chunk, which only describes a serpent.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from starter.agent_metrics import aggregate

RUNS = {
    "agent-tfidf-dev": {
        "grades": {"SH001": 2, "SH002": 2, "SH003": 2, "SH004": 2, "SH005": 2,
                   "SH006": 2, "SH007": 0, "SH008": 2, "SH009": 0, "SH010": 2,
                   "SH011": 2, "SH012": 2, "SH013": 0, "SH014": 2, "SH015": 0},
        "unsupported_citations": set(),
    },
    "agent-bm25-dev-final": {
        "grades": {"SH001": 2, "SH002": 2, "SH003": 2, "SH004": 2, "SH005": 2,
                   "SH006": 2, "SH007": 2, "SH008": 2, "SH009": 1, "SH010": 2,
                   "SH011": 2, "SH012": 2, "SH013": 1, "SH014": 2, "SH015": 0},
        "unsupported_citations": set(),
    },
    "agent-tfidf-test": {
        "grades": {"D7T001": 0, "D7T002": 0, "D7T003": 0, "D7T004": 2, "D7T005": 0},
        "unsupported_citations": {"D7T002"},
    },
    "agent-bm25-test": {
        "grades": {"D7T001": 0, "D7T002": 0, "D7T003": 0, "D7T004": 0, "D7T005": 2},
        "unsupported_citations": {"D7T002"},
    },
}


def main() -> None:
    for run_name, config in RUNS.items():
        run_dir = ROOT / "submission/results" / run_name
        cases = json.loads((run_dir / "cases.json").read_text(encoding="utf-8"))
        traces = json.loads((run_dir / "traces.json").read_text(encoding="utf-8"))
        by_id = {t["query_id"]: t for t in traces}
        reviews = []
        for case in cases:
            cid, trace = case["id"], by_id[case["id"]]
            support = [cid not in config["unsupported_citations"] for _ in trace.get("citations", [])]
            review = {"query_id": cid, "citation_support": support}
            if case["answerable"]:
                review["answer_grade"] = config["grades"][cid]
            else:
                review["abstention_correct"] = trace["status"] == "abstain"
            reviews.append(review)
        (run_dir / "reviews.json").write_text(json.dumps(reviews, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        split = cases[0]["split"]
        summary = {"split": split, **aggregate(cases, traces, reviews),
                   "review_note": "Assistant-prepared content review; student should confirm labels before submission."}
        (run_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(run_name, json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
