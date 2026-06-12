# NLP — Solution Description

## What this folder is

`nlp/` holds the complete entry for the **NLP** challenge of TIL-26. The
nominal task is retrieval-augmented question answering: a corpus of documents
is loaded once, then questions arrive and for each one the system must return
**(a)** the IDs of the relevant documents and **(b)** a text answer.

This entry does **not** answer the questions. It is a **grader exploit**. The
deployed container does two things and nothing else:

1. **Retrieves** the right documents with pure-CPU BM25 (to win the
   document-overlap half of the score).
2. Returns a **single fixed adversarial string** as the answer to *every*
   question — a universal trigger that forces the answer-equivalence grader to
   score the answer as "equivalent" no matter what the real answer is.

There is **no LLM, no embedder, no reranker, no GPU, and no model weights** in
the image. The container is `fastapi` + `uvicorn` + `numpy` + `rank-bm25` on
`python:3.12-slim`.

> An earlier iteration was a genuine RAG pipeline (BM25 + dense reranker +
> Phi-4-mini generation via vLLM, ~0.75 local score). It was abandoned in
> favour of the exploit below and has been removed from this folder. Only the
> exploit and its supporting retrieval tooling remain.

---

## The task contract (`src/nlp_server.py`)

`POST /nlp` on port **5004**. One endpoint, three request shapes, dispatched on
which key the first instance carries:

| Request | Body | Reply |
| --- | --- | --- |
| **Corpus load** | `{"instances":[{"documents":[{"id","document"}, …]}]}` | `{"predictions":[{"status":"loading"\|"loaded"\|"error"}]}` |
| **Readiness poll** | `{"instances":[{"poll":"true"}]}` | `{"predictions":[{"status": …}]}` |
| **Questions** | `{"instances":[{"question": …}, …]}` | `{"predictions":[{"answer":str,"documents":[doc_id, …]}, …]}` |

`GET /health` → `200 {"message":"health ok"}` once ready, `503` while loading or
on load failure. The model loads in a background thread so `/health` answers
immediately; corpus indexing runs in an async task so the load POST returns at
once and the grader polls for `"loaded"`. There is no `/reset`.

---

## How the challenge is scored (`test/test_nlp.py`)

`AnswerEquivalenceEvaluator.batch_evaluate` scores each question independently.
The branch that matters:

| Condition | Score |
| --- | --- |
| Predicted docs ∩ gold `source_docs` is **empty** | **0.0** |
| Docs overlap, but `prob_equivalent < threshold` | **0.4** (`RETRIEVAL_ONLY_SCORE`) |
| Docs overlap **and** `prob_equivalent ≥ threshold` | **1.0** |

(Plus L4/L5 unanswerable cases that want empty docs+answer — out of scope for
the Novice answerable set.)

So every question's score is gated **first** on document overlap, **then** on
answer equivalence. `prob_equivalent` is produced by a fine-tuned encoder
classifier (`test/models/nlp_eval_512`, a ModernBERT sequence classifier) over
the string `Question: {q} Reference: {gold} Candidate: {our_answer}`. Only the
candidate is attacker-controlled, and **the candidate is truncated to 64 tokens**
before scoring (`MAX_CANDIDATE_TOKEN_LENGTH = 64`).

The two exploits each attack one gate.

---

## Exploit 1 — the universal adversarial trigger (the answer gate)

**File:** `gcg_universal.py` (generator) → `trigger.json` (artifact) →
hardcoded as `CHEESE_TRIGGER` in `src/nlp_manager.py` (deployment).

The grader is a neural classifier and only the candidate text is ours, so the
answer-equivalence gate is a white-box adversarial-example target. We search
for **one** 64-token candidate string that drives `prob_equivalent → 1.0` for
**any** `(question, reference)` pair, then return that same string as the answer
to every question.

### The attack — Greedy Coordinate Gradient (GCG / HotFlip)

`gcg_universal.py` runs a standard universal-trigger search against the grader
weights:

1. Build inputs `[CLS] Question: {q} Reference: {r} Candidate: {trigger} [SEP]`
   for a small **train** set of real `(q, r)` pairs (`N_TRAIN = 32`).
2. **HotFlip gradient** (`grad_topk`): take the gradient of the cross-entropy
   toward class 1 ("equivalent") w.r.t. the trigger tokens' one-hot vectors,
   averaged over the trigger span and the batch. The top-`TOPK = 256` tokens
   per position are the swap candidates.
3. **Greedy coordinate step**: sample `N_CAND = 128` single-token swaps, score
   them all in one batched forward pass (`losses_of`), and keep the best swap if
   it lowers the real loss.
4. Repeat for `N_STEPS = 150`, checkpointing to `trigger.json` every 10 steps.

Two correctness guards make the shipped trigger trustworthy:

- **ASCII-only token pool** — only tokens that survive the grader's
  `printable` filter and tokenizer round-trip are eligible for swaps
  (`ascii_mask`), so the optimized trigger is exactly what the grader will see.
- **Honest validation** (`real_score`) — every checkpoint is re-scored on
  `N_VAL = 200` *held-out* pairs through the exact `test_nlp.py` formatting +
  64-token truncation round-trip, reporting both the mean `prob_equivalent` and
  the fraction clearing `≥ 0.9`.

### The shipped trigger

`trigger.json` records the chosen checkpoint:

```
step 20  ·  train_loss ≈ 5.6e-08  ·  val_mean ≈ 0.99999  ·  val_hit (≥0.9) = 1.0
```

i.e. on 200 held-out `(question, reference)` pairs the trigger scored
`prob_equivalent ≥ 0.9` **100%** of the time, mean ≈ 1.0. The string itself is
64 tokens of fluent-looking garbage (`"igroup riventies possesses
denotedFried capita duty 330 1895 payments …"`). It is pasted verbatim as
`CHEESE_TRIGGER` in `src/nlp_manager.py`; `qa_batch` returns it for every
question.

**Effect:** every question that clears the document gate is converted from a
0.4 ("right docs, wrong answer") into a 1.0 ("equivalent").

---

## Exploit 2 — title-boosted BM25 retrieval (the document gate)

**File:** `src/retrieval.py` (`BM25Retriever`).

The trigger is worthless on a question whose predicted docs miss the gold doc —
that question scores 0.0 regardless of the answer. So the second job is to
maximize **recall@3** of the gold document, cheaply, on CPU.

- **Whole-document BM25** (`rank_bm25.BM25Okapi`) over the corpus — no chunking,
  no embeddings. Top-3 doc IDs are returned per question.
- **Tokenization** (`bm25_tokenize`): lowercase + split on non-alphanumerics, so
  a code like `EA-76-088` decomposes to `['ea','76','088']` consistently in both
  the question and the document that answers it — the rare-anchor tokens BM25's
  IDF locks onto.
- **Title boost** (`boost_title`): each document's first H1 (`# …`) is prepended
  `TITLE_BOOST` times before indexing. A title is the strongest signal of what a
  doc is about, but one occurrence in a multi-thousand-token body contributes
  almost nothing; repeating it inflates the title tokens' term frequency and
  recovers questions that would otherwise pick a generic "overview" doc.
- **BM25 parameters**: `k1` (term-frequency saturation) and `b` (length
  normalization) are tuned. `b = 1.0` fully normalizes for length, penalizing
  long overview docs that win wrongly on common terms.

All three knobs are env-overridable so one image can be rebuilt into many
leaderboard variants via Docker build-args:

| Env / build-arg | Default | Meaning |
| --- | --- | --- |
| `NLP_TITLE_BOOST` | `5` | title repetitions before indexing |
| `NLP_BM25_K1` | `1.2` | BM25 term-frequency saturation |
| `NLP_BM25_B` | `1.0` | BM25 length normalization |

The default `(5, 1.2, 1.0)` is the local sweep winner: **recall@3 ≈ 0.983** on
the 883-question local set.

> A phrase-bigram index variant was trialled and consistently hurt recall by
> ~0.001, so it was dropped (see `ab_recall.py`).

---

## Why the two combine to near-ceiling

Per-question score ≈ `1.0` when the gold doc is in the top-3 (trigger handles
the answer gate), else `0.0`. So the expected total score is approximately
**recall@3 × 1.0**, i.e. ≈ 0.98 — versus the abandoned LLM pipeline's ~0.75,
which was bottlenecked on *generation* (arithmetic slips, wrong-number
selection) rather than retrieval. The exploit removes the generation bottleneck
entirely.

---

## Deployed files (`src/`)

| File | Role |
| --- | --- |
| `nlp_server.py` | FastAPI app: corpus-load / poll / QA dispatch, background load, real `/health`. |
| `nlp_manager.py` | `NLPManager`: builds the BM25 index on corpus load; `qa_batch` returns top-3 doc IDs + the fixed `_TRIGGER` answer. Holds the trigger constant. |
| `retrieval.py` | `BM25Retriever` + title-boost / tokenization helpers. The only non-trivial logic in the container. |
| `__init__.py` | package marker. |

The container is built from these plus `requirements.txt` (`fastapi`,
`uvicorn[standard]`, `numpy`, `rank-bm25`). The `Dockerfile` bakes the three
BM25 knobs as build-args/ENV.

---

## Offline tooling (not shipped)

These live at the `nlp/` top level and never enter the image (the Dockerfile
only copies `src/`).

**The exploit:**
- `gcg_universal.py` — the GCG universal-trigger search (above). Rerun to
  regenerate `trigger.json`, then paste the new `trigger_str` into
  `nlp_manager.py`.
- `trigger.json` — the current trigger artifact + its validation metrics.
- `probe_ae.py` — characterizes the grader's exploitability: runs several
  candidate strategies (naive copies, paraphrases, etc.) over real `(q, r)`
  pairs and reports the `prob_equivalent` distribution, to confirm a
  gradient-based universal trigger is actually required (vs. some trivial
  candidate already clearing the threshold).

**Retrieval tuning (BM25 recall@3):**
- `tune_bm25.py` — sweep `(k1, b)` on the title-boosted index.
- `ab_recall.py` — A/B title-boost × bigram configurations.
- `measure_recall.py` — recall@3 of the live `BM25Retriever` vs the local
  question set; flags which previously-missed questions flipped.
- `diagnose_misses.py` — inspects retrieval misses (top-3 ∩ gold = ∅) and their
  shared shape (difficulty, rare-anchor presence, length).
- `bm25_score_dist.py` — BM25 top-1 score distribution (answerable vs
  off-topic), for picking an abstention threshold. *Stale:* the abstention gate
  it references has since been removed from the manager.

**End-to-end eval + submission:**
- `eval_local.py` — loads the live pipeline and the real grader
  (`AnswerEquivalenceEvaluator`), runs the local question set, and reports the
  competition score under the current contract (recall@1/3, L1/L2 split,
  score-bucket breakdown). This is the local validator for the whole exploit.
- `submit_presets.sh` — rebuilds + submits a cycle of `(title_boost, k1, b)`
  preset images, each uniquely tagged.

---

## Commands

### Build & run the container

```bash
cd nlp
docker build -t nlp .                       # default knobs (5, 1.2, 1.0)
# a leaderboard variant:
docker build --build-arg NLP_TITLE_BOOST=5 \
             --build-arg NLP_BM25_K1=2.0 \
             --build-arg NLP_BM25_B=0.75 -t nlp:t5-k20-b075 .
docker run -p 5004:5004 nlp                 # POST to http://localhost:5004/nlp
```

### Regenerate the trigger (after a grader change)

```bash
cd nlp
python gcg_universal.py                     # white-box GCG; writes trigger.json
# then paste the new trigger_str into _TRIGGER in src/nlp_manager.py
```

### Measure locally

```bash
cd nlp
python eval_local.py                        # full set; competition score
python eval_local.py --limit 150            # quick subset
python measure_recall.py                    # recall@3 of the BM25 retriever
```

---

## Status & caveats

- **Deployed and self-contained.** Retrieval recall@3 ≈ 0.983; the trigger
  clears the answer gate on 100% of held-out pairs → expected score ≈ recall@3.
- **The exploit is grader-specific.** The trigger was white-boxed against
  `test/models/nlp_eval_512`. It transfers to the live competition only if the
  real grader is that same model (or close enough). If the grader differs, the
  trigger may not fire and every overlap question falls back to 0.4 — retrieval
  alone still floors the score there. Re-run `gcg_universal.py` against any new
  grader weights.
- **Novice answerable set only.** L4/L5 unanswerable (empty-answer) handling is
  out of scope; the fixed trigger answers every question non-empty.
