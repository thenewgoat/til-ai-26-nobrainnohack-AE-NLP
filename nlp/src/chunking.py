"""Header-aware Markdown chunking for the RAG corpus.

Documents are structured Markdown. We split on headers, then pack each
section's body into overlapping token windows, prepending the document title
and section header to every chunk so the embedder, reranker, and LLM all see
the chunk's provenance.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class Chunk:
    """One retrievable unit of text and the index of its parent document."""

    text: str
    doc_index: int


def _split_sections(doc: str):
    """Return (title, [(section_header, body_text), ...]) for a Markdown doc."""
    lines = doc.splitlines()

    title = ""
    for ln in lines:
        m = _HEADER_RE.match(ln.strip())
        if m and len(m.group(1)) == 1:
            title = m.group(2).strip()
            break

    segments = []
    cur_header = ""
    cur_body: list[str] = []
    for ln in lines:
        m = _HEADER_RE.match(ln.strip())
        if m:
            if cur_body:
                segments.append((cur_header, "\n".join(cur_body).strip()))
                cur_body = []
            cur_header = m.group(2).strip()
        else:
            cur_body.append(ln)
    if cur_body:
        segments.append((cur_header, "\n".join(cur_body).strip()))

    return title, segments


def chunk_document(doc, doc_index, tokenizer,
                   target_tokens=350, overlap_tokens=64):
    """Split one Markdown document into overlapping, header-prefixed Chunks."""
    title, segments = _split_sections(doc)
    chunks: list[Chunk] = []

    for header, body in segments:
        if not body:
            continue
        prefix = " > ".join(p for p in (title, header) if p)
        token_ids = tokenizer(body, add_special_tokens=False)["input_ids"]
        if not token_ids:
            continue
        step = max(1, target_tokens - overlap_tokens)
        for start in range(0, len(token_ids), step):
            piece = token_ids[start:start + target_tokens]
            if not piece:
                break
            body_text = tokenizer.decode(piece, skip_special_tokens=True)
            text = f"{prefix}\n{body_text}" if prefix else body_text
            chunks.append(Chunk(text=text, doc_index=doc_index))
            if start + target_tokens >= len(token_ids):
                break

    # A document with no usable Markdown structure still yields one chunk.
    if not chunks and doc.strip():
        chunks.append(Chunk(text=doc.strip()[:2000], doc_index=doc_index))

    return chunks


def chunk_corpus(documents, tokenizer, target_tokens=350, overlap_tokens=64):
    """Chunk every document in the corpus, preserving parent-document indices."""
    out: list[Chunk] = []
    for i, doc in enumerate(documents):
        out.extend(chunk_document(doc, i, tokenizer, target_tokens, overlap_tokens))
    return out
