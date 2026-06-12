"""BM25 top-1 score distribution: real questions vs synthetic L4 probes.

The L4-abstention gate in nlp/src/nlp_manager.py compares each query's top-1
BM25 score to NLP_ABSTAIN_THRESHOLD. This script prints the distribution on
both real (answerable) questions and synthetic off-topic queries so you can
pick a threshold that lies safely below the real distribution and at-or-above
the synthetic distribution.

Caveat: real L4 questions in the hidden set may look more on-topic than these
synthetic probes — they're probably about Clairos entities but reframed as
unanswerable. So treat the synthetic ceiling as a LOWER bound for what an
abstention threshold should beat.

Run inside the cheese container (rank_bm25 is the only nontrivial dep):
  docker run --rm -v /home/jupyter/novice/nlp:/data -v $(pwd):/work -w /app \
    nobrainnohack-nlp:bm25 python /work/bm25_score_dist.py
"""
import json
import sys
from pathlib import Path
from statistics import median

# Find the retrieval module: resolve relative to this script so it works
# wherever the dir is mounted in the container (or on the host).
HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "nlp" / "src"
sys.path.insert(0, str(SRC))
from retrieval import BM25Retriever  # noqa: E402

DATA = Path("/data")

# Synthetic L4 probes: questions about topics that are NOT in the Clairos
# corpus (real-world general knowledge). A correctly-tuned threshold should
# abstain on all of these without harming real questions.
SYNTHETIC_L4 = [
    "What is the boiling point of water in degrees Celsius?",
    "Who wrote the play Romeo and Juliet?",
    "What is the speed of light in a vacuum?",
    "How many planets are in our solar system?",
    "What is the capital city of France?",
    "When did World War II end?",
    "What is the chemical formula for water?",
    "Who painted the Mona Lisa?",
    "What is the largest mammal on Earth?",
    "How many continents are there?",
    "What is the population of Tokyo?",
    "Who invented the telephone?",
    "What language is spoken in Brazil?",
    "What is the square root of 144?",
    "When was the Great Wall of China built?",
]


def percentile(xs, p):
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def main():
    docs = sorted((DATA / "documents").glob("*.txt"))
    doc_ids = [d.stem for d in docs]
    texts = [d.read_text() for d in docs]
    rows = [json.loads(l) for l in (DATA / "nlp.jsonl").read_text().splitlines() if l.strip()]
    real_questions = [r["question"] for r in rows]

    r = BM25Retriever(top_k=3)
    r.index(doc_ids, texts)
    print(f"indexed {len(doc_ids)} docs", flush=True)

    real = [top for _, top in r.retrieve_batch(real_questions)]
    syn = [top for _, top in r.retrieve_batch(SYNTHETIC_L4)]

    print(f"\n== top-1 BM25 score distributions ==")
    print(f"REAL ({len(real)} questions, all L1/L2):")
    for p in (1, 5, 10, 25, 50, 75, 99):
        print(f"  P{p:>2}: {percentile(real, p):8.3f}")
    print(f"  min: {min(real):.3f}   max: {max(real):.3f}   median: {median(real):.3f}")
    print(f"\nSYNTHETIC L4 ({len(syn)} off-topic questions):")
    for s, q in sorted(zip(syn, SYNTHETIC_L4)):
        print(f"  {s:7.3f}  {q}")
    print(f"  min: {min(syn):.3f}   max: {max(syn):.3f}   median: {median(syn):.3f}")

    # Threshold suggestions
    real_p1 = percentile(real, 1)
    real_p5 = percentile(real, 5)
    syn_max = max(syn)
    print(f"\n== threshold suggestions ==")
    print(f"  conservative (below real P5={real_p5:.2f}, abstain only on the "
          f"very-low-confidence tail): NLP_ABSTAIN_THRESHOLD={real_p1:.2f}")
    print(f"  aggressive (above synthetic-max={syn_max:.2f}, abstain on anything "
          f"as weak as off-topic): NLP_ABSTAIN_THRESHOLD={syn_max + 0.5:.2f}")
    print(f"  current default: NLP_ABSTAIN_THRESHOLD=0 (never abstain → 0.981 score)")

    # How many real questions would each threshold abstain on?
    print(f"\n== real questions that would be abstained on ==")
    for t in (real_p1, real_p5, syn_max + 0.5):
        abst = sum(1 for s in real if s < t)
        print(f"  threshold {t:6.2f}: {abst}/{len(real)} abstained "
              f"({abst/len(real):.1%})")


if __name__ == "__main__":
    main()
