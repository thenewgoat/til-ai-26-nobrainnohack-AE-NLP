"""Large-batch throughput benchmark for the NLP RAG QA pipeline.

eval_local.py runs batch-4, which cannot reveal gains from a larger KV cache /
more concurrent sequences. This feeds one big batch through qa_batch and reports
questions/sec, isolating retrieval time from generation time.

Usage:
  python bench_throughput.py            # default 64-question batch
  python bench_throughput.py --batch 128
"""
import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import nlp_manager

DATA_DIR = "/home/jupyter/novice/nlp"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=64,
                    help="number of questions in the single timed batch")
    args = ap.parse_args()

    nlp_manager.MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

    doc_paths = sorted(glob.glob(f"{DATA_DIR}/documents/*.txt"))
    documents = [{"id": os.path.basename(p)[:-4], "document": open(p).read()}
                 for p in doc_paths]
    rows = [json.loads(ln) for ln in open(f"{DATA_DIR}/nlp.jsonl") if ln.strip()]

    # Cycle the question pool up to the requested batch size.
    questions = [rows[i % len(rows)]["question"] for i in range(args.batch)]

    print(f"Loading pipeline; indexing {len(documents)} documents ...",
          flush=True)
    manager = nlp_manager.NLPManager()
    manager.load_corpus(documents)

    # Warm-up so the timed run excludes lazy CUDA init / compilation.
    manager.qa_batch(questions[:4])

    n = len(questions)

    # Phase-split timing: retrieval (embed + BM25 + rerank) vs generation.
    t0 = time.time()
    retrieved = manager.retriever.retrieve_batch(questions)
    t_retrieve = time.time() - t0

    convs = [manager._build_messages(q, c)
             for q, c in zip(questions, retrieved) if c]
    t1 = time.time()
    manager.llm.chat(convs, manager.sampling, use_tqdm=False)
    t_generate = time.time() - t1

    total = t_retrieve + t_generate
    print("\n==== THROUGHPUT ====")
    print(f"batch size:        {n}")
    print(f"retrieval:         {t_retrieve:.2f} s  ({t_retrieve / n * 1000:.0f} ms/q)")
    print(f"generation:        {t_generate:.2f} s  ({t_generate / n * 1000:.0f} ms/q)")
    print(f"total:             {total:.2f} s")
    print(f"throughput:        {n / total:.2f} questions/s")
    print(f"per-question:      {total / n * 1000:.1f} ms")


if __name__ == "__main__":
    main()
