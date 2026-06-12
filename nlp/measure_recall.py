"""Measure recall@3 with the current BM25Retriever against local 883 questions.

Used to verify title-weighting + bigram tweaks before rebuilding the container.
Also reports which of the previous 17 misses are now hits (and any new misses
introduced — defensive check that the tweaks don't regress easy questions).
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "nlp" / "src"
sys.path.insert(0, str(SRC))
from retrieval import BM25Retriever  # noqa: E402

DATA = Path("/data")

# The 17 previously-missed question keys (from diagnose_misses.py output).
PREV_MISSES = {28, 50, 148, 161, 213, 269, 275, 280, 295, 354, 427,
               488, 596, 649, 825, 847, 880}


def main():
    docs = sorted((DATA / "documents").glob("*.txt"))
    doc_ids = [d.stem for d in docs]
    texts = [d.read_text() for d in docs]
    rows = [json.loads(l) for l in (DATA / "nlp.jsonl").read_text().splitlines() if l.strip()]
    questions = [r["question"] for r in rows]

    r = BM25Retriever(top_k=3)
    r.index(doc_ids, texts)
    preds = r.retrieve_batch(questions)

    hits = 0
    new_misses = []
    recovered_prev_misses = []
    still_missing_prev = []
    for row, pred in zip(rows, preds):
        gold = set(row["source_docs"])
        if gold & set(pred):
            hits += 1
            if row["key"] in PREV_MISSES:
                recovered_prev_misses.append(row["key"])
        else:
            if row["key"] not in PREV_MISSES:
                new_misses.append(row["key"])
            else:
                still_missing_prev.append(row["key"])

    n = len(rows)
    recall = hits / n
    print(f"recall@3 = {recall:.4f}  ({hits}/{n})")
    print(f"  vs previous (plain BM25): 0.9807 (866/883)")
    print(f"  delta = {recall - 0.9807:+.4f}")
    print()
    print(f"recovered  previous misses: {len(recovered_prev_misses)}/17  "
          f"{sorted(recovered_prev_misses)}")
    print(f"still miss previous misses: {len(still_missing_prev)}/17  "
          f"{sorted(still_missing_prev)}")
    print(f"NEW misses introduced:      {len(new_misses)}  "
          f"{sorted(new_misses)}")


if __name__ == "__main__":
    main()
