"""Run the seven supplied questions and score retrieval against a small gold set."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from rag import MiniRAG


BASE_DIR = Path(__file__).resolve().parent
GOLD = [
    {"status": "answered", "docs": {"DOC-02"}, "facts": ["24"]},
    {"status": "answered", "docs": {"DOC-03"}, "facts": ["48"]},
    {"status": "answered", "docs": {"DOC-04", "DOC-05"}, "facts": ["1", "стойк"]},
    {"status": "answered", "docs": {"DOC-04"}, "facts": ["1500"]},
    {"status": "answered", "docs": {"DOC-05"}, "facts": ["18:00", "администр"]},
    {"status": "clarification_needed", "docs": set(), "facts": ["уточните"]},
    {"status": "insufficient_data", "docs": {"DOC-06"}, "facts": ["недостаточно данных"]},
]


def load_questions() -> list[str]:
    lines = (BASE_DIR / "questions.txt").read_text(encoding="utf-8-sig").splitlines()
    return [line.split(". ", 1)[1] for line in lines if line[:1].isdigit()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("chunk", "llm"), default="chunk")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rag = MiniRAG(BASE_DIR / "documents")
    rows = []
    for number, (question, gold) in enumerate(zip(load_questions(), GOLD), 1):
        result = rag.answer(question, args.mode)
        actual_docs = {source["doc_id"] for source in result["sources"]}
        answer_text = result["answer"].lower()
        checks = {
            "status_correct": result["status"] == gold["status"],
            "documents_correct": actual_docs == gold["docs"],
            "answer_facts_correct": all(fact.lower() in answer_text for fact in gold["facts"]),
            "quotes_verbatim": all(
                source["text"] in next((BASE_DIR / "documents").glob(f"*{source['doc_id'][4:]}*.txt")).read_text(encoding="utf-8-sig")
                for source in result["sources"]
            ),
        }
        rows.append({"number": number, **result, "evaluation": checks, "passed": all(checks.values())})
    output = args.output or BASE_DIR / f"results_{args.mode}.csv"
    fieldnames = [
        "number", "question", "mode", "status", "answer", "source_doc_ids", "quotes",
        "status_correct", "documents_correct", "answer_facts_correct", "quotes_verbatim", "passed",
    ]
    with output.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "number": row["number"],
                "question": row["question"],
                "mode": row["mode"],
                "status": row["status"],
                "answer": row["answer"],
                "source_doc_ids": "; ".join(source["doc_id"] for source in row["sources"]),
                "quotes": " || ".join(source["text"] for source in row["sources"]),
                **row["evaluation"],
                "passed": row["passed"],
            })
    passed = sum(row["passed"] for row in rows)
    print(f"{args.mode}: {passed}/{len(rows)} passed; results: {output}")


if __name__ == "__main__":
    main()
