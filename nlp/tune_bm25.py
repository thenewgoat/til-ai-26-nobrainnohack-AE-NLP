"""Sweep BM25Okapi (k1, b) parameters with the title-boosted index.

k1 controls term-frequency saturation (higher → more reward for repeated terms).
b controls length normalization (1=full normalize, 0=ignore length).

Default in rank_bm25 is k1=1.5, b=0.75 — that gave us 0.9819 with title-boost.
Sweep around that to see if there's further free recall on top.
"""
import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

DATA = Path("/data")
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_H1_RE = re.compile(r"^\s*#\s+(.+?)\s*$", re.MULTILINE)
TITLE_BOOST = 5


def extract_title(text):
    m = _H1_RE.search(text)
    return m.group(1).strip() if m else ""


def tokenize(text):
    return _TOKEN_RE.findall(text.lower())


def boost(text):
    title = extract_title(text)
    return f"{(title + ' ') * TITLE_BOOST}\n{text}" if title else text


def main():
    docs = sorted((DATA / "documents").glob("*.txt"))
    doc_ids = [d.stem for d in docs]
    texts = [d.read_text() for d in docs]
    rows = [json.loads(l) for l in (DATA / "nlp.jsonl").read_text().splitlines() if l.strip()]
    tokenized = [tokenize(boost(t)) for t in texts]
    print(f"{len(doc_ids)} docs, {len(rows)} questions, title boost={TITLE_BOOST}\n",
          flush=True)

    K1S = (0.5, 0.8, 1.2, 1.5, 2.0, 3.0)
    BS = (0.0, 0.25, 0.5, 0.75, 1.0)
    print(f"{'k1\\b':>6}" + "".join(f"{b:>9.2f}" for b in BS))
    best = (0.0, None, None)
    for k1 in K1S:
        row = [f"{k1:>6.2f}"]
        for b in BS:
            bm25 = BM25Okapi(tokenized, k1=k1, b=b)
            hits = 0
            for r in rows:
                scores = bm25.get_scores(tokenize(r["question"]))
                top3 = sorted(range(len(scores)), key=lambda i: scores[i],
                              reverse=True)[:3]
                if set(r["source_docs"]) & {doc_ids[i] for i in top3}:
                    hits += 1
            recall = hits / len(rows)
            if recall > best[0]:
                best = (recall, k1, b)
            row.append(f"{recall:>9.4f}")
        print("".join(row), flush=True)

    print(f"\nbest: recall={best[0]:.4f}  k1={best[1]}  b={best[2]}")
    print(f"baseline (title=5, k1=1.5, b=0.75): 0.9819")


if __name__ == "__main__":
    main()
