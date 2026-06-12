"""Characterize the answer-equivalence (AE) grader to gauge how exploitable it is.

The AE model (`test/models/nlp_eval_512`, ModernBertForSequenceClassification)
scores `Question: {q} Reference: {r} Candidate: {c}` and emits prob_equivalent.
Only the candidate `c` is attacker-controlled. This probe runs several candidate
strategies over real (question, reference) pairs and reports the prob_equivalent
distribution per strategy, so we know whether a trivial candidate already clears
the 0.9 threshold or a gradient-based universal trigger is required.
"""
import json
import random
from pathlib import Path
from string import printable

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

AE_MODEL = Path(__file__).resolve().parents[1] / "test/models/nlp_eval_512"
NLP_JSONL = Path("/home/jupyter/novice/nlp/nlp.jsonl")
MAX_CANDIDATE_TOKEN_LENGTH = 64   # matches test_nlp.py _format_input
MAX_LENGTH = 512
THRESHOLD = 0.9
N_PAIRS = 250                     # subset of (q, r) pairs for a fast probe

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok = AutoTokenizer.from_pretrained(str(AE_MODEL))
model = AutoModelForSequenceClassification.from_pretrained(str(AE_MODEL)).to(device).eval()


def format_input(question: str, reference: str, candidate: str) -> str:
    """Exact replica of test_nlp.py AnswerEquivalenceEvaluator._format_input."""
    _printable = "".join(filter(lambda x: x in printable, candidate))
    tokens = tok.tokenize(
        _printable, max_length=MAX_CANDIDATE_TOKEN_LENGTH, truncation=True)
    reconstructed = tok.convert_tokens_to_string(tokens)
    return f"Question: {question} Reference: {reference} Candidate: {reconstructed}"


@torch.no_grad()
def prob_equivalent(triples, batch_size=32) -> torch.Tensor:
    """Return prob_equivalent (class 1) for a list of (q, r, c) triples."""
    out = []
    for i in range(0, len(triples), batch_size):
        texts = [format_input(q, r, c) for q, r, c in triples[i:i + batch_size]]
        enc = tok(texts, max_length=MAX_LENGTH, padding="longest",
                  truncation=True, return_tensors="pt").to(device)
        probs = F.softmax(model(**enc).logits, dim=-1)
        out.append(probs[:, 1].cpu())
    return torch.cat(out)


def report(name: str, probs: torch.Tensor) -> None:
    hit = (probs >= THRESHOLD).float().mean().item()
    print(f"  {name:<22} mean={probs.mean():.3f}  median={probs.median():.3f}  "
          f"min={probs.min():.3f}  max={probs.max():.3f}  "
          f">=0.9: {hit:6.1%}")


def main() -> None:
    rows = [json.loads(line) for line in NLP_JSONL.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r.get("answer")]
    random.seed(0)
    random.shuffle(rows)
    rows = rows[:N_PAIRS]
    qr = [(r["question"], r["answer"]) for r in rows]
    print(f"AE model: {AE_MODEL.name} on {device} | {len(qr)} (question, reference) pairs\n")

    vocab_words = "report data system value number record total amount the is a of".split()

    strategies = {
        "gold (=reference)":  lambda q, r: r,
        "empty":              lambda q, r: "",
        "wrong fixed":        lambda q, r: "the city of Singapore",
        "echo question":      lambda q, r: q,
        "affirmative phrase": lambda q, r: "Yes that is correct, the answer is equivalent and the same.",
        "random words":       lambda q, r: " ".join(random.choices(vocab_words, k=20)),
    }

    for name, fn in strategies.items():
        triples = [(q, r, fn(q, r)) for q, r in qr]
        report(name, prob_equivalent(triples))

    # Per-token logit bias: feed each single vocab token as the whole candidate,
    # averaged over pairs — surfaces tokens the classifier reacts to.
    print("\nTop single-token candidates by mean prob_equivalent:")
    sample = qr[:60]
    scores = []
    common = tok.convert_ids_to_tokens(range(1000, 6000))
    for piece in tqdm(common[:1500], desc="token sweep"):
        word = tok.convert_tokens_to_string([piece]).strip()
        if not word or not word.isascii():
            continue
        triples = [(q, r, word) for q, r in sample]
        scores.append((prob_equivalent(triples).mean().item(), word))
    scores.sort(reverse=True)
    for s, w in scores[:15]:
        print(f"  {s:.3f}  {w!r}")


if __name__ == "__main__":
    main()
