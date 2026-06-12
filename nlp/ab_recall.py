"""Sweep title-boost x bigrams configurations to find the actual winner.

The combined (title+bigrams) implementation showed a wash (delta 0.0). This
script monkey-patches the retrieval to test each variant individually so we
can see whether either change alone helps, or whether the wash is genuine
(i.e., pure BM25 is just a local optimum on this corpus).
"""
import json
import re
import sys
from pathlib import Path

from rank_bm25 import BM25Okapi

DATA = Path("/data")
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_H1_RE = re.compile(r"^\s*#\s+(.+?)\s*$", re.MULTILINE)


def extract_title(text):
    m = _H1_RE.search(text)
    return m.group(1).strip() if m else ""


def make_tokenize(use_bigrams):
    def f(text):
        toks = _TOKEN_RE.findall(text.lower())
        return toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])] if use_bigrams else toks
    return f


def make_boost(title_boost):
    def f(text):
        title = extract_title(text)
        return f"{(title + ' ') * title_boost}\n{text}" if title and title_boost > 0 else text
    return f


def main():
    docs = sorted((DATA / "documents").glob("*.txt"))
    doc_ids = [d.stem for d in docs]
    texts = [d.read_text() for d in docs]
    rows = [json.loads(l) for l in (DATA / "nlp.jsonl").read_text().splitlines() if l.strip()]
    print(f"{len(doc_ids)} docs, {len(rows)} questions\n", flush=True)
    print(f"{'config':<28} recall@3")
    print("-" * 45)

    for title_boost in (0, 3, 5, 10):
        for use_bigrams in (False, True):
            boost = make_boost(title_boost)
            tok = make_tokenize(use_bigrams)
            bm25 = BM25Okapi([tok(boost(t)) for t in texts])
            hits = 0
            for row in rows:
                scores = bm25.get_scores(tok(row["question"]))
                top3 = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]
                if set(row["source_docs"]) & {doc_ids[i] for i in top3}:
                    hits += 1
            label = f"title={title_boost}  bigrams={'ON ' if use_bigrams else 'OFF'}"
            print(f"{label:<28} {hits/len(rows):.4f}  ({hits}/{len(rows)})",
                  flush=True)


if __name__ == "__main__":
    main()
