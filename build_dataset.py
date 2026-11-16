"""Adapt the prepared Day 6 corpus to the Day 7 benchmark and chunk schema."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path


FRESH_TEST = [
    {"id": "D7T001", "split": "test", "question": "In 'The Red-Headed League', what was the name of Jabez Wilson's assistant?", "expected_answer": "Vincent Spaulding.", "gold_source_id": "pg1661", "gold_page": "pg1661_story_02", "evidence_quote": "His name is Vincent Spaulding", "filter": None},
    {"id": "D7T002", "split": "test", "question": "In 'The Adventure of the Speckled Band', what kind of snake did Holmes identify?", "expected_answer": "A swamp adder.", "gold_source_id": "pg1661", "gold_page": "pg1661_story_08", "evidence_quote": "It is a swamp adder!", "filter": None},
    {"id": "D7T003", "split": "test", "question": "In 'The Adventure of the Copper Beeches', what annual salary did Jephro Rucastle offer Violet Hunter?", "expected_answer": "One hundred pounds a year.", "gold_source_id": "pg1661", "gold_page": "pg1661_story_12", "evidence_quote": "Your salary with me, madam, would commence at £ 100 a year.", "filter": None},
    {"id": "D7T004", "split": "test", "question": "In 'The Adventure of the Engineer's Thumb', which finger did Victor Hatherley lose?", "expected_answer": "His thumb.", "gold_source_id": "pg1661", "gold_page": "pg1661_story_09", "evidence_quote": "Hatherley’s thumb", "filter": None},
    {"id": "D7T005", "split": "test", "question": "In 'The Adventure of the Missing Three-Quarter', what was the missing Cambridge player's name?", "expected_answer": "Godfrey Staunton.", "gold_source_id": "pg108", "gold_page": "pg108_story_11", "evidence_quote": "Godfrey Staunton", "filter": None},
]

UNANSWERABLE = [
    {"id": "D7U001", "split": "dev", "question": "What was Sherlock Holmes's telephone number at Baker Street?", "answerable": False, "expected_behavior": "search_and_abstain", "absence_reason": "The corpus contains no telephone number for Holmes; searched all 37 processed story units for telephone-number terms."},
    {"id": "D7U002", "split": "dev", "question": "What was the exact date of birth of Dr. Watson's wife?", "answerable": False, "expected_behavior": "search_and_abstain", "absence_reason": "The corpus does not give Watson's wife's date of birth; searched all 37 processed story units for birth-date and date-of-birth terms."},
    {"id": "D7U003", "split": "test", "question": "Which mobile phone number could Watson use to call 221B Baker Street?", "answerable": False, "expected_behavior": "search_and_abstain", "absence_reason": "The corpus gives no mobile phone number; searched all 37 processed story units for mobile-phone terms."},
    {"id": "D7U004", "split": "test", "question": "What was Dr. Watson's email address?", "answerable": False, "expected_behavior": "search_and_abstain", "absence_reason": "The corpus gives no email address for Watson; searched all 37 processed story units for email-address terms."},
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--day6-root", type=Path, default=Path(r"C:\AI_K20\Lab\K5-L3-Day06-2A202603731-DataIngestionEngineering"))
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent.parent / "data")
    args = parser.parse_args()
    root = args.day6_root.resolve()
    starter = root / "submission/starter"
    manifest = json.loads((starter / "data/manifest.json").read_text(encoding="utf-8"))
    old = json.loads((starter / "data/benchmark.json").read_text(encoding="utf-8"))["questions"]
    sys.path.insert(0, str(starter))
    from src.chunking import by_paragraph
    from src.ingestion import normalize_text
    from src.models import DocumentPage

    source_titles = {s["source_id"]: s["title"] for s in manifest["sources"]}
    chunks = []
    output_corpus = args.output / "corpus"
    output_corpus.mkdir(parents=True, exist_ok=True)
    for unit in manifest["units"]:
        text = normalize_text((starter / unit["source_path"]).read_text(encoding="utf-8"))
        shutil.copy2(starter / unit["source_path"], output_corpus / f"{unit['page_id']}.txt")
        page = DocumentPage(unit["source_id"], source_titles[unit["source_id"]], unit["page_id"], text,
                            {"section": unit["title"], "source_id": unit["source_id"]})
        for c in by_paragraph(page, max_size=1000):
            chunks.append({"id": c.chunk_id, "text": c.text, "source_id": c.document_id,
                           "source": c.source, "page": str(c.page), "metadata": c.metadata})

    # Retain a fixed dev slice, then add two explicit no-evidence cases.
    dev_answerable = []
    for q in [x for x in old if x["split"] == "dev"][:15]:
        dev_answerable.append({"id": q["id"], "split": "dev", "question": q["question"],
                               "expected_answer": q["expected_answer"], "answerable": True,
                               "gold": [{"source_id": q["gold_source_id"], "page": q["gold_page"], "evidence": q["evidence_quote"]}],
                               "filter": q.get("filter")})
    fresh = []
    for q in FRESH_TEST:
        fresh.append({"id": q["id"], "split": q["split"], "question": q["question"],
                      "expected_answer": q["expected_answer"], "answerable": True,
                      "gold": [{"source_id": q["gold_source_id"], "page": q["gold_page"], "evidence": q["evidence_quote"]}],
                      "filter": q.get("filter")})
    questions = dev_answerable + UNANSWERABLE + fresh
    for q in questions:
        for g in q.get("gold", []):
            norm = lambda value: " ".join(str(value).split())
            assert any(c["source_id"] == g["source_id"] and c["page"] == g["page"] and norm(g["evidence"]) in norm(c["text"]) for c in chunks), (q["id"], g)
    # Check the claimed absences in the complete copied source corpus before recording them.
    corpus_text = "\n".join(p.read_text(encoding="utf-8").casefold() for p in output_corpus.glob("*.txt"))
    assert all(term not in corpus_text for term in ("telephone number", "date of birth", "mobile phone", "email address"))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "questions.json").write_text(json.dumps(questions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    cases = [{"id": q["id"], "split": q["split"], "answerable": q["answerable"]} for q in questions]
    (args.output / "cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"source_units": len(manifest["units"]), "chunks": len(chunks),
                      "questions": len(questions), "dev": sum(q["split"] == "dev" for q in questions),
                      "test": sum(q["split"] == "test" for q in questions),
                      "dev_answerable": sum(q["split"] == "dev" and q["answerable"] for q in questions),
                      "test_answerable": sum(q["split"] == "test" and q["answerable"] for q in questions)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
