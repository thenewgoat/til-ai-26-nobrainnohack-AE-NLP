"""Doc-level BM25 retrieval with title-weighted indexing.

A pure-CPU retriever that scores every corpus document for each query using
BM25 (rank_bm25). No embedder, no GPU.

One tweak past plain bag-of-words BM25: prepend each doc's H1 title
`TITLE_BOOST` times before indexing. A doc's title is the strongest signal
of what the doc is about, but a single occurrence in a 5k-token body gets
drowned out. Repeating it inflates the title tokens' TF — helps recover
common-anchor questions that would otherwise pick an "overview" doc.

(Phrase bigrams were trialled and consistently hurt recall by ~0.001 on the
local set, so they were dropped. See nlp_cheese/ab_recall.py.)
"""
from __future__ import annotations

import os
import re

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-z0-9]+")
# Markdown H1 line: '#' followed by whitespace and the title text. The \s+
# between '#' and the title prevents '##' (H2) from matching.
_H1_RE = re.compile(r"^\s*#\s+(.+?)\s*$", re.MULTILINE)

# Tunables (env-overridable so a single image can be built into multiple
# leaderboard variants via Docker build-args; defaults are the local sweep
# winner: title=5, k1=1.2, b=1.0 → recall@3 0.9830).
TITLE_BOOST = int(os.environ.get("NLP_TITLE_BOOST", "5"))
BM25_K1 = float(os.environ.get("NLP_BM25_K1", "1.2"))
BM25_B = float(os.environ.get("NLP_BM25_B", "1.0"))


def extract_title(text: str) -> str:
    """Return the first H1 (# heading) in a Markdown doc, or '' if none."""
    m = _H1_RE.search(text)
    return m.group(1).strip() if m else ""


def bm25_tokenize(text: str) -> list[str]:
    """Lowercase + split on non-alphanumerics. Decomposes codes like
    'EA-76-088' into ['ea','76','088'] consistently, so a question and the
    document that answers it share those rare tokens."""
    return _TOKEN_RE.findall(text.lower())


def boost_title(text: str) -> str:
    """Prepend the doc's H1 title TITLE_BOOST times for BM25 indexing.

    Without this the title is one occurrence among thousands of body tokens
    and contributes almost nothing to a doc's score for title-relevant
    queries. Repeating it inflates the title tokens' TF.
    """
    title = extract_title(text)
    return f"{(title + ' ') * TITLE_BOOST}\n{text}" if title else text


class BM25Retriever:
    """Whole-document BM25 with title-weighted indexing.

    Usage:
        r = BM25Retriever(top_k=3)
        r.index(doc_ids, texts)
        r.retrieve_batch(questions)  # -> [[doc_id, ...], ...]
    """

    def __init__(self, top_k: int = 3):
        self.top_k = top_k
        self.doc_ids: list[str] = []
        self._bm25: BM25Okapi | None = None

    def index(self, doc_ids, texts) -> None:
        """Build the BM25 index over the corpus documents (title-weighted)."""
        self.doc_ids = list(doc_ids)
        if not texts:
            self._bm25 = None
            return
        # k1 + b set from env (defaults: 1.2 / 1.0). Lower k1 reduces
        # over-reward for repeated terms in overview-style docs; b=1.0 fully
        # normalizes for doc length, penalizing long "overview" docs that win
        # wrongly on common-term questions. Swept in nlp_cheese/tune_bm25.py.
        self._bm25 = BM25Okapi(
            [bm25_tokenize(boost_title(t)) for t in texts],
            k1=BM25_K1, b=BM25_B)

    def retrieve_batch(self, questions) -> list[list[str]]:
        """Return the top-K doc ids per question (BM25 ranking).

        Questions are NOT title-boosted (they have no title); the index does.
        """
        if self._bm25 is None or not self.doc_ids:
            return [[] for _ in questions]
        n = len(self.doc_ids)
        k = min(self.top_k, n)
        out = []
        for q in questions:
            scores = self._bm25.get_scores(bm25_tokenize(q))
            idx = sorted(range(n), key=lambda i: scores[i], reverse=True)[:k]
            out.append([self.doc_ids[i] for i in idx])
        return out
