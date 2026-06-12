"""GCG universal adversarial trigger against the AE grader (`nlp_eval_512`).

Searches for a single candidate token sequence that drives prob_equivalent >= 0.9
for ANY (question, reference) pair. White-box: gradients are taken w.r.t. the
trigger token one-hots (HotFlip), top-k swaps are sampled, the best real-loss
swap is kept (Greedy Coordinate Gradient).

The inner loop uses a manually constructed token layout for speed; every reported
score is re-validated through the exact test_nlp.py `_format_input` round-trip so
the trigger we ship is the trigger the real grader sees.
"""
import json
import random
from pathlib import Path
from string import printable

import torch
import torch.nn.functional as F
from tqdm import trange
from transformers import AutoModelForSequenceClassification, AutoTokenizer

AE_MODEL = Path(__file__).resolve().parents[1] / "test/models/nlp_eval_512"
NLP_JSONL = Path("/home/jupyter/novice/nlp/nlp.jsonl")
OUT = Path(__file__).resolve().parent / "trigger.json"

TRIGGER_LEN = 64        # candidate is truncated to 64 tokens by the grader
N_TRAIN = 32            # (q, r) pairs the trigger is optimized over
N_VAL = 200             # held-out pairs for honest generalization measurement
TOPK = 256              # candidate tokens per position from the gradient
N_CAND = 128            # swaps evaluated per GCG step
N_STEPS = 150
CHUNK = 256             # sequences per forward pass when scoring candidates
SEED = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok = AutoTokenizer.from_pretrained(str(AE_MODEL))
model = AutoModelForSequenceClassification.from_pretrained(str(AE_MODEL)).to(device).eval()
for p in model.parameters():
    p.requires_grad_(False)
embed = model.get_input_embeddings()
V = embed.weight.shape[0]

CLS, SEP = tok.cls_token_id, tok.sep_token_id


def head_ids(question: str, reference: str) -> list[int]:
    """Token ids for '[CLS] Question: {q} Reference: {r} Candidate:' (no trigger)."""
    text = f"Question: {question} Reference: {reference} Candidate:"
    return [CLS] + tok(text, add_special_tokens=False).input_ids


# ---- ASCII-only token pool: triggers must survive the printable filter + round-trip
ascii_tokens = []
for tid in range(V):
    s = tok.convert_tokens_to_string([tok.convert_ids_to_tokens(tid)])
    if s and all(c in printable for c in s):
        ascii_tokens.append(tid)
ascii_mask = torch.zeros(V, dtype=torch.bool, device=device)
ascii_mask[torch.tensor(ascii_tokens, device=device)] = True


def build_batch(pairs, trigger):
    """Return padded input_ids + attention_mask for [head | trigger | SEP]."""
    seqs = [h + trigger + [SEP] for h, _ in pairs]
    m = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), m), tok.pad_token_id, dtype=torch.long)
    att = torch.zeros((len(seqs), m), dtype=torch.long)
    tpos = []  # (row, start) of the trigger span per sequence
    for i, (h, _) in enumerate(pairs):
        s = h + trigger + [SEP]
        ids[i, :len(s)] = torch.tensor(s)
        att[i, :len(s)] = 1
        tpos.append(len(h))
    return ids.to(device), att.to(device), tpos


@torch.no_grad()
def losses_of(pairs, triggers) -> torch.Tensor:
    """Mean CE loss toward class 1 for each trigger, scored over `pairs`.

    All trigger x pair sequences are forwarded together in CHUNK-sized batches.
    """
    seqs = [h + trg + [SEP] for trg in triggers for h, _ in pairs]
    ce = []
    for i in range(0, len(seqs), CHUNK):
        chunk = seqs[i:i + CHUNK]
        m = max(len(s) for s in chunk)
        ids = torch.full((len(chunk), m), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(chunk), m), dtype=torch.long)
        for j, s in enumerate(chunk):
            ids[j, :len(s)] = torch.tensor(s)
            att[j, :len(s)] = 1
        logits = model(input_ids=ids.to(device), attention_mask=att.to(device)).logits
        tgt = torch.ones(len(chunk), dtype=torch.long, device=device)
        ce.append(F.cross_entropy(logits, tgt, reduction="none").cpu())
    return torch.cat(ce).view(len(triggers), len(pairs)).mean(dim=1)


def grad_topk(pairs, trigger):
    """HotFlip gradient: top-k token candidates per trigger position."""
    ids, att, tpos = build_batch(pairs, trigger)
    onehot = F.one_hot(ids, V).float().requires_grad_(True)
    inp = onehot @ embed.weight
    logits = model(inputs_embeds=inp, attention_mask=att).logits
    F.cross_entropy(logits, torch.ones(len(pairs), dtype=torch.long, device=device)).backward()
    g = onehot.grad  # [B, T, V]
    # Average the gradient over the trigger span across the batch.
    acc = torch.zeros(TRIGGER_LEN, V, device=device)
    for i, start in enumerate(tpos):
        acc += g[i, start:start + TRIGGER_LEN, :]
    acc[:, ~ascii_mask] = float("inf")  # forbid non-ascii swaps
    return (-acc).topk(TOPK, dim=1).indices  # [T, TOPK]


def real_score(trigger):
    """Validate the trigger through the exact grader pipeline (string round-trip)."""
    cand = tok.convert_tokens_to_string(tok.convert_ids_to_tokens(trigger))
    triples, refs = [], []
    for q, r in val:
        _p = "".join(c for c in cand if c in printable)
        toks = tok.tokenize(_p, max_length=TRIGGER_LEN, truncation=True)
        recon = tok.convert_tokens_to_string(toks)
        triples.append(f"Question: {q} Reference: {r} Candidate: {recon}")
    with torch.no_grad():
        out = []
        for i in range(0, len(triples), 64):
            enc = tok(triples[i:i + 64], max_length=512, padding="longest",
                      truncation=True, return_tensors="pt").to(device)
            out.append(F.softmax(model(**enc).logits, dim=-1)[:, 1].cpu())
    p = torch.cat(out)
    return p.mean().item(), (p >= 0.9).float().mean().item()


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    rows = [json.loads(l) for l in NLP_JSONL.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("answer")]
    random.shuffle(rows)
    global val
    train = [(head_ids(r["question"], r["answer"]), None) for r in rows[:N_TRAIN]]
    val = [(r["question"], r["answer"]) for r in rows[N_TRAIN:N_TRAIN + N_VAL]]

    trigger = random.choices(ascii_tokens, k=TRIGGER_LEN)
    best = losses_of(train, [trigger])[0].item()
    print(f"init loss {best:.4f}", flush=True)

    for step in trange(N_STEPS, desc="GCG"):
        cand_tok = grad_topk(train, trigger)
        cands = []
        for _ in range(N_CAND):
            pos = random.randrange(TRIGGER_LEN)
            t = trigger.copy()
            t[pos] = cand_tok[pos, random.randrange(TOPK)].item()
            cands.append(t)
        losses = losses_of(train, cands)
        j = int(losses.argmin())
        if losses[j].item() < best:
            best, trigger = losses[j].item(), cands[j]
        if step % 10 == 0 or step == N_STEPS - 1:
            vm, vh = real_score(trigger)
            print(f"step {step:3d}  train_loss {best:.4f}  "
                  f"val_mean {vm:.3f}  val>=0.9 {vh:.1%}", flush=True)
            OUT.write_text(json.dumps({
                "trigger_ids": trigger,
                "trigger_str": tok.convert_tokens_to_string(tok.convert_ids_to_tokens(trigger)),
                "step": step, "train_loss": best, "val_mean": vm, "val_hit": vh}, indent=2))

    print(f"\nsaved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
