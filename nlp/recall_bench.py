"""Measure recall@3 and timing for retrieval variants — reranker vs not.

recall@3 here = fraction of questions whose top-3 returned document ids contain
at least one gold source_doc. That is exactly the document-overlap gate in
test_nlp.py, so recall@3 IS the cheese pipeline's score ceiling.

Run inside the cheese container:
  docker run --rm --gpus all -v /home/jupyter/novice/nlp:/data \
    -v $(pwd):/work -w /app nobrainnohack-nlp:cheese python /work/recall_bench.py
"""
import json
import sys
import time
from pathlib import Path

# Import the working-copy retrieval code (mounted at /work), NOT the stale
# /app/src baked into the image.
sys.path.insert(0, "/work/src")
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

from chunking import chunk_corpus
from retrieval import Retriever

MODELS = "/app/models"
DATA = Path("/data")


def resolve(repo_id):
    import glob, os
    snaps = sorted(glob.glob(os.path.join(
        MODELS, "models--" + repo_id.replace("/", "--"), "snapshots", "*")))
    return snaps[-1]


def top3_docs(chunks, doc_ids):
    out = []
    for c in chunks:
        d = doc_ids[c.doc_index]
        if d not in out:
            out.append(d)
    return out[:3]


def recall_at3(results, doc_ids, gold):
    hit = 0
    for chunks, g in zip(results, gold):
        if set(top3_docs(chunks, doc_ids)) & set(g):
            hit += 1
    return hit / len(results)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fp16 = {"torch_dtype": torch.float16} if device == "cuda" else {}

    # Corpus + questions.
    docs = sorted((DATA / "documents").glob("*.txt"))
    doc_ids = [d.stem for d in docs]
    texts = [d.read_text() for d in docs]
    rows = [json.loads(l) for l in (DATA / "nlp.jsonl").read_text().splitlines() if l.strip()]
    questions = [r["question"] for r in rows]
    gold = [r["source_docs"] for r in rows]
    print(f"{len(texts)} documents | {len(questions)} questions", flush=True)

    # FULL (dense+BM25+reranker) was already measured: recall@3=0.9807,
    # 725 ms/q. Only the NO-RERANK variant is run here — the reranker model
    # is not even loaded, so startup is faster too.
    embedder = SentenceTransformer(resolve("BAAI/bge-large-en-v1.5"),
                                   device=device, model_kwargs=fp16)

    t = time.time()
    chunks = chunk_corpus(texts, embedder.tokenizer)
    print(f"chunking: {len(chunks)} chunks in {time.time()-t:.1f}s", flush=True)

    r = Retriever(embedder, reranker=None)
    t = time.time()
    r.index(chunks)
    print(f"index (embed {len(chunks)} chunks + BM25): "
          f"{time.time()-t:.1f}s", flush=True)

    t = time.time()
    results = r.retrieve_batch(questions)
    dt = time.time() - t
    rec = recall_at3(results, doc_ids, gold)
    print(f"  NO-RERANK (dense+BM25 RRF)  recall@3={rec:.4f}  "
          f"retrieval={dt:.1f}s ({dt/len(questions)*1000:.0f} ms/q)", flush=True)
    print("  FULL (measured earlier)     recall@3=0.9807  "
          "retrieval=640.5s (725 ms/q)", flush=True)


if __name__ == "__main__":
    main()
