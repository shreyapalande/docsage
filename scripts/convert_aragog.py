"""
Converts ARAGOG benchmark.json into DocsAge golden_dataset.json format.
Usage: python scripts/convert_aragog.py
"""

import json
from pathlib import Path


def assign_difficulty(question: str) -> str:
    word_count = len(question.split())
    if word_count < 15:
        return "easy"
    elif word_count <= 25:
        return "medium"
    return "hard"


def main() -> None:
    src = Path("data/benchmark.json")
    dst = Path("eval/golden_dataset.json")

    raw = json.loads(src.read_text(encoding="utf-8"))
    questions = raw["questions"]
    answers = raw["ground_truths"]

    assert len(questions) == len(answers), "Question/answer count mismatch"

    pairs = []
    for i, (q, a) in enumerate(zip(questions, answers), start=1):
        pairs.append({
            "id": f"q{i:03d}",
            "question": q.strip(),
            "reference_answer": a.strip(),
            "expected_doc_ids": [],
            "difficulty": assign_difficulty(q.strip()),
        })

    counts = {"easy": 0, "medium": 0, "hard": 0}
    for p in pairs:
        counts[p["difficulty"]] += 1

    dataset = {
        "version": "1.0",
        "description": "Golden Q&A pairs converted from ARAGOG benchmark for DocsAge eval harness",
        "pairs": pairs,
    }

    dst.write_text(json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Written {len(pairs)} pairs to {dst}")
    print(f"  easy: {counts['easy']}  medium: {counts['medium']}  hard: {counts['hard']}")


if __name__ == "__main__":
    main()
