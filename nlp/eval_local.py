"""In-process evaluation harness for the NLP RAG QA pipeline.

Loads the pipeline and the ModernBERT answer-equivalence scorer, runs the
local question set, and reports the competition score under the CURRENT
scoring contract (test/test_nlp.py):

  - no overlap between predicted docs and gold source_docs -> 0.0
  - overlap but answer not equivalent (prob < 0.9)          -> 0.4
  - overlap and answer equivalent                           -> 1.0

The reported score is the mean over all questions. Also reports the L1/L2
split, the score-bucket breakdown, and retrieval recall@1/3.

Usage:
  python eval_local.py            # full question set
  python eval_local.py --limit 150  # quick subset for tuning iterations
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
# test_nlp.py and its AnswerEquivalenceEvaluator live in the repo test/ folder.
sys.path.insert(0, "/home/jupyter/til-ai-26/test")

import nlp_manager

DATA_DIR = "/home/jupyter/novice/nlp"
EVAL_MODEL = "/home/jupyter/til-ai-26/test/models/nlp_eval_512"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate only the first N questions")
    args = ap.parse_args()

    # Use the host model cache; the container uses /app/models.
    nlp_manager.MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

    # Load documents as {"id","document"} dicts, matching the task contract.
    doc_paths = sorted(glob.glob(f"{DATA_DIR}/documents/*.txt"))
    documents = [{"id": os.path.basename(p)[:-4], "document": open(p).read()}
                 for p in doc_paths]

    rows = [json.loads(ln) for ln in open(f"{DATA_DIR}/nlp.jsonl") if ln.strip()]
    if args.limit:
        rows = rows[:args.limit]

    print(f"Loading pipeline; indexing {len(documents)} documents ...",
          flush=True)
    manager = nlp_manager.NLPManager()
    manager.load_corpus(documents)

    from test_nlp import AnswerEquivalenceEvaluator
    scorer = AnswerEquivalenceEvaluator(
        model_path=EVAL_MODEL, threshold=0.9, max_length=512)

    import time
    preds = []
    recall_hits = {1: 0, 3: 0}
    batch = 4  # mirrors the request batch size used by test_nlp.py
    t0 = time.time()
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        preds.extend(manager.qa_batch([r["question"] for r in chunk]))
        if (i // batch) % 6 == 0:
            print(f"  {min(i + batch, len(rows))}/{len(rows)} answered",
                  flush=True)
    qa_secs = time.time() - t0
    for r, pred in zip(rows, preds):
        pdocs = pred["documents"]
        src = r["source_docs"][0]
        if pdocs and src == pdocs[0]:
            recall_hits[1] += 1
        if src in pdocs[:3]:
            recall_hits[3] += 1

    # 5-tuple scoring: (gold_docs, pred_docs[:3], question, gold, candidate)
    data = [(r["source_docs"], p["documents"][:3], r["question"],
             r["answer"] or "", p["answer"])
            for r, p in zip(rows, preds)]
    results = scorer.batch_evaluate(data)
    summary = scorer.aggregate_score(results)

    n = len(rows)
    buckets = {1.0: 0, 0.4: 0, 0.0: 0}
    by_diff = {}
    for r, res in zip(rows, results):
        buckets[res.score] = buckets.get(res.score, 0) + 1
        d = r["difficulty"]
        agg = by_diff.setdefault(d, [0.0, 0])
        agg[0] += res.score
        agg[1] += 1

    print("\n==== RESULTS ====")
    print(f"SCORE (mean): {summary['equiv_rate']:.4f}   (n={n})")
    print(f"QA time: {qa_secs:.1f}s total, {qa_secs / n:.2f}s/question")
    for d, (tot, cnt) in sorted(by_diff.items()):
        if cnt:
            print(f"  {d}: {tot / cnt:.4f}  ({cnt} q)")
    print(f"buckets: correct(1.0)={buckets.get(1.0, 0)}  "
          f"retrieval-only(0.4)={buckets.get(0.4, 0)}  "
          f"miss(0.0)={buckets.get(0.0, 0)}")
    print(f"retrieval recall@1: {recall_hits[1] / n:.4f}")
    print(f"retrieval recall@3: {recall_hits[3] / n:.4f}")

    detail = [
        {"question": r["question"], "difficulty": r["difficulty"],
         "gold": r["answer"], "source_docs": r["source_docs"],
         "pred_answer": p["answer"], "pred_docs": p["documents"],
         "score": res.score, "equivalent": res.equivalent,
         "prob_equivalent": res.prob_equivalent}
        for r, p, res in zip(rows, preds, results)
    ]
    out = os.path.join(os.path.dirname(__file__), "eval_results.json")
    with open(out, "w") as f:
        json.dump({"score": summary["equiv_rate"], "n": n, "detail": detail}, f,
                  indent=2)
    print(f"\nSaved detailed results to {out}")


if __name__ == "__main__":
    main()
