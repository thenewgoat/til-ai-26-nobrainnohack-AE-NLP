"""Diagnose the retrieval misses from the last til-test run.

A "miss" = pred["documents"][:3] doesn't intersect gt["source_docs"] → 0.0
on the document-overlap gate, regardless of the (perfect) trigger answer.

Reads:
  /home/jupyter/nobrainnohack/nlp_results.json   (server predictions)
  /home/jupyter/novice/nlp/nlp.jsonl             (ground truth)

Prints each missed question with its gold + predicted docs, plus simple
pattern stats (difficulty, has-rare-token, question length) to suggest
whether the misses share a fixable shape.
"""
import json
import re
from collections import Counter
from pathlib import Path

RESULTS = Path("/home/jupyter/nobrainnohack/nlp_results.json")
JSONL = Path("/home/jupyter/novice/nlp/nlp.jsonl")

# A "rare anchor" heuristic: ALL-CAPS multi-letter tokens, alphanumeric codes
# like EA-76-088, or TitleCase multi-word names. The kind of tokens BM25's IDF
# locks onto. Questions WITHOUT one are the BM25 weak spot.
ANCHOR_RE = re.compile(r"\b[A-Z]{2,}\b|\b[A-Z]+-?\d+[-\d]*\b|\b[A-Z][a-z]+(?:\s[A-Z][a-z]+){1,}\b")


def has_anchor(q: str) -> bool:
    return bool(ANCHOR_RE.search(q))


def main():
    preds = json.loads(RESULTS.read_text())
    rows = [json.loads(l) for l in JSONL.read_text().splitlines() if l.strip()]
    assert len(preds) == len(rows), f"len mismatch {len(preds)} vs {len(rows)}"

    misses = []
    by_diff = Counter()
    miss_by_diff = Counter()
    miss_with_anchor = 0
    miss_without_anchor = 0
    hits_with_anchor = 0
    hits_without_anchor = 0
    for p, g in zip(preds, rows):
        pred_docs = p.get("documents", [])[:3]
        gold = g["source_docs"]
        hit = bool(set(pred_docs) & set(gold))
        anchor = has_anchor(g["question"])
        by_diff[g["difficulty"]] += 1
        if hit:
            if anchor: hits_with_anchor += 1
            else: hits_without_anchor += 1
        else:
            miss_by_diff[g["difficulty"]] += 1
            if anchor: miss_with_anchor += 1
            else: miss_without_anchor += 1
            misses.append((g, p))

    print(f"== summary == {len(misses)}/{len(rows)} missed (recall@3 = "
          f"{1 - len(misses)/len(rows):.4f})\n")
    print("miss rate by difficulty:")
    for d, n in sorted(by_diff.items()):
        m = miss_by_diff[d]
        print(f"  {d}: {m}/{n} miss  ({m/n:.1%})")
    print("\nanchor-token presence in questions:")
    print(f"  miss with anchor:    {miss_with_anchor}")
    print(f"  miss WITHOUT anchor: {miss_without_anchor}")
    print(f"  hit  with anchor:    {hits_with_anchor}")
    print(f"  hit  WITHOUT anchor: {hits_without_anchor}")
    if (miss_with_anchor + miss_without_anchor) and (hits_with_anchor + hits_without_anchor):
        m_rate_no_anchor = miss_without_anchor / max(1, miss_without_anchor + hits_without_anchor)
        m_rate_anchor = miss_with_anchor / max(1, miss_with_anchor + hits_with_anchor)
        print(f"  → miss rate WITHOUT anchor: {m_rate_no_anchor:.1%}  "
              f"vs WITH anchor: {m_rate_anchor:.1%}")

    print(f"\n== the {len(misses)} misses ==\n")
    for i, (g, p) in enumerate(misses, 1):
        q = g["question"]
        anchors = ANCHOR_RE.findall(q)
        print(f"[{i:2d}] {g['difficulty']}  key={g['key']}  "
              f"anchors={anchors if anchors else 'NONE'}")
        print(f"     Q: {q}")
        print(f"     gold docs: {g['source_docs']}  gold answer: {g['answer']!r}")
        print(f"     pred docs: {p.get('documents', [])[:3]}")
        print()


if __name__ == "__main__":
    main()
