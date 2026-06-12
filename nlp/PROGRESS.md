# nlp_shizhen — Progress, Measurements & Research Notes

_Running log. Most recent first._

## Status (2026-05-19) — inference-speed research + n-gram trial

Scope correction: this is the **Novice track — L1 and L2 questions only**. L3
(cross-doc), L4/L5 (unanswerable) are NOT in scope. All abstention / empty-answer
work is dropped.

### Speed: throughput profiling (decision: focus speed, accept 0.761 accuracy)

Large-batch throughput, batch-64 single `qa_batch` (`bench_throughput.py`):

| Phase | Time | per-q |
|---|---|---|
| Retrieval (embed + BM25 + rerank) | 22.5 s | 352 ms |
| Generation (vLLM) | 40.3 s | 629 ms |
| **Total** | **62.8 s** | **981 ms** (1.02 q/s) |

Batch-64 is only ~1.4× faster per-q than batch-4 — weak scaling. Generation is
64% of the time, retrieval 36%.

**Why the weak scaling — KV-cache bound.** vLLM startup logs (gpu_mem 0.70):
`Available KV cache memory: 2.45 GiB; GPU KV cache size: 19,960 tokens;
Maximum concurrency for 3,072 tokens/request: 6.50x`. Only ~6.5 sequences can
run concurrently — a "batch of 64" really executes ~7-wide with churn.

**CORRECTION** to an earlier note that dismissed INT4: the claim "batch-64 runs
without KV pressure" was wrong. The workload IS KV-bound above batch ~6.
Therefore the KV-cache levers ARE worthwhile:
- **Raise `gpu_memory_utilization`** 0.70 → ~0.85 (free, zero accuracy risk):
  the helper models only need ~2.5 GB of the non-vLLM pool, so ~2 GB is idle.
- **Lower `max_model_len`** 3072 → ~2304 (real prompts ≈1930 tok): concurrency
  is computed per max_model_len, so this alone raises it ~33%.
- **INT4 quantization**: Phi-4-mini weights 7.15 GB → ~2.3 GB frees ~4.8 GB →
  KV cache ~3× → concurrency ~6.5x → ~19x. The real large-batch lever (NOT for
  per-token speed — Turing has no Marlin — but for concurrency).

**Caveat that gates all of the above:** `test_nlp.py` hardcodes `BATCH_SIZE=4`.
4 < 6.5, so at batch-4 there is NO KV pressure and these levers give ~nothing.
They only pay off if the real grader sends batches >6. Open question.

**CUDA graphs** (`enforce_eager=False`, env `NLP_ENFORCE_EAGER`): separate
~1.1–1.3× decode-loop win, zero accuracy risk. Benchmark repeatedly blocked by
`ae`-track GPU contention; unmeasured.

### Prompt tweaks (anti-abstention + natural-rounding) — TRIALLED, REVERTED

| Run | Score (150-q) | L1 | L2 | correct |
|---|---|---|---|---|
| Baseline | 0.7610 | 0.8095 | 0.6489 | 93 |
| + 2 prompt rules | 0.7490 | 0.7981 | 0.6356 | 90 |

Net regression. Diff vs baseline: **9 regressed, 6 improved**. Most flips are
unrelated to the two new rules (`4-0` → `Aye, Aye, Aye, Aye`; `10` → `Twelve`;
a literal `FINAL ANSWER` emitted as the answer) — i.e. adding rules caused
diffuse output drift, not a targeted fix. The 2 abstention cases targeted were
not fixed. Reverted.

Lesson: greedy vLLM is deterministic for identical input (the n-gram run
reproduced 0.7610 exactly), so the eval is a reliable measurement tool — but
small prompt edits cause large, unpredictable, net-random output drift. Prompt
micro-tweaking is not a reliable lever; future generation changes should be
structural (different model / decoding strategy) and measured on a larger set.

Full question set: nlp.jsonl has **883 questions (592 L1, 291 L2)**; the 150-q
subset is just the first 150. A full-set eval (~20 min) gives a far more stable
score for judging small changes.

Infra kept (score-neutral): NLP_GPU_MEM_FRACTION and NLP_MAX_NUM_SEQS env
overrides; max_num_seqs now defaults to 64 (was vLLM's 256, which OOM'd the
sampler warmup when the T4 was shared).

### n-gram speculative decoding — TRIALLED, REVERTED (no gain)

| Run | Score (150-q) | QA time | s/question |
|---|---|---|---|
| Baseline | 0.7610 | 199.0 s | 1.33 |
| + ngram speculative | 0.7610 | 199.6 s | 1.33 |

Score identical (lossless under greedy, as expected). **Speed unchanged.**
Reasons: (1) vLLM disables its async scheduler when ngram spec-decode is on —
the two roughly cancel; (2) speculative decoding's benefit shrinks with batch
size, and the eval runs batch-4 (real workload wants *larger* batches); (3) the
measured time also includes retrieval. Reverted — keeping it would hurt
throughput at large batch. Config already had chunked prefill + prefix caching
+ async scheduling on by default.

### Baseline failure analysis (150-q, current pipeline)

Score 0.7610 — L1 0.8095 (105 q), L2 0.6489 (45 q). Buckets: correct 93,
retrieval-only(0.4) 53, miss 4. Recall@3 0.973.

The 25 L2 failures, categorised from `eval_results.json`:
- **~9 wrong-number-picked** — model selects the wrong figure from context
  (48 vs 37, 44 vs 28, 19 vs 3, 35M vs 75M). A grounding/comprehension error;
  Program-of-Thought does NOT fix this (wrong inputs).
- **~4 arithmetic slips** — right inputs, wrong operation/conversion (Zonnon
  `2/38` not converted to 5.3%; reactors computed for 1 not 2). PoT helps here.
- **2 wrongful abstention** — model says "cannot be determined" though the
  answer exists. On Novice L1/L2 every question is answerable → a one-line
  anti-abstention prompt rule fixes these. Low risk.
- **2 over-precision** — `5 years 5 months` vs `5 years`, `14.72` vs
  `14 months`. Prompt rule: round naturally.
- **~6 phrasing / completeness / wrong-content** — harder, mixed.

## Status (2026-05-18, overnight autonomous session)

New RAG QA pipeline built, contract-migrated, and submitted. Currently blocked
on GPU (in use by a teammate) — research/prep mode.

### Submissions to Vertex AI (`nobrainnohack-nlp`, task `nlp`)
1. **#1** — baseline prompt, **old I/O contract**. Superseded (would score ~0
   under the updated scoring).
2. **#2** — contract-migrated image. **FAILED** — `docker push` denied,
   `Unauthenticated`: `~/.docker/config.json` had no credential helper for
   `asia-southeast1-docker.pkg.dev`. Fixed with
   `gcloud auth configure-docker asia-southeast1-docker.pkg.dev`.
3. **#3 — LIVE** — contract-migrated + revised prompt + answer normalizer
   (`nobrainnohack-nlp:shizhen`, digest `sha256:a7cb6c77…`). Push + Vertex
   registration succeeded. Local 150-q score ~0.769 (normalizer effect not yet
   re-measured — eval OOM'd when the GPU was taken).

Policy: rebuild + resubmit after every measured improvement.

## Measurements (local 150-question subset)

| Version | Scoring | Score |
|---|---|---|
| Baseline prompt | old all-or-nothing | 0.593 |
| Revised prompt (terse, exact-copy) | old all-or-nothing | 0.667 |
| Revised prompt | **new contract** (5-tuple) | **0.769** |
| + answer normalizer | new contract | _pending GPU_ |

0.769 breakdown: correct (1.0) = 95, retrieval-only (0.4) = 51, miss (0.0) = 4.
L1 mean 0.817, L2 mean 0.658, retrieval recall@3 ≈ 0.97.

## The task contract changed mid-development (handled)

`test/test_nlp.py` was updated; pipeline migrated to match (commit `327d872`):
corpus docs are `{"id","document"}` dicts; load/poll replies are
`{"predictions":[{"status":...}]}`; each QA prediction is
`{"answer", "documents":[doc_id,…]}`. Scoring: no doc overlap → 0.0; right doc,
wrong answer → 0.4; right doc + equivalent answer → 1.0; equivalence threshold
0.9; candidate truncated to 64 tokens. Hidden set adds L4/L5 (unanswerable).

## Failure analysis — the 51 "retrieval-only / 0.4" questions

- **19 — numeric/reasoning errors** (biggest single lever): the model reasons
  but bungles arithmetic (`73` vs `37` years, `1.23M` vs `7.8M`, `Q4 77` vs
  `Q4 78`). Research confirms this is the classic small-LLM failure: correct
  rationale, wrong calculation.
- **25 — casing / phrasing / completeness**: some fixed by the normalizer
  (`korren - 8` → `korren-8`); some are the terse prompt now *over-truncating*
  needed detail (`19` vs `19 of 42`; dropped `47 to 39` score).
- **4 — verbose** (full sentences instead of phrases).
- **3 — just under the 0.9 equivalence threshold** (hard to chase).
- **4 retrieval misses (0.0)** — retrieval is near-ceiling, leave alone.

## Research findings

- Reranking + hybrid retrieval = highest-ROI retrieval techniques — already in
  the pipeline; retrieval recall@3 ≈ 0.97, near-solved.
- Our bottleneck is **generation**, not retrieval (opposite of the typical RAG
  case where ~73% of failures are retrieval).
- **Small-LLM arithmetic**: CoT helps but small models still make calculation
  slips. Strongest fixes: **Program-of-Thought** (model emits the calculation,
  host evaluates it) and `Phi-4-mini-reasoning` (math-tuned, same 3.8 B size,
  already downloaded to `models/` for trialling).
- **Conciseness**: answers should be concise but *complete* — the current terse
  prompt slightly over-truncates; aim for "shortest phrase that includes every
  part the question asks for."

## Prioritised next experiments (when GPU is free)

1. Re-measure the answer normalizer on 150-q (confirm submission #3's gain).
2. **Prompt rebalance**: "shortest phrase that still contains every requested
   part" — fixes over-truncation without losing terseness. Low risk, no latency
   cost. Measure; if better, rebuild + resubmit.
3. **L2 reasoning**: trial `Phi-4-mini-reasoning` as the generator (math-tuned),
   and/or a Program-of-Thought step. Targets the 19 numeric errors — the largest
   remaining block. Measure carefully (latency + L1 regression risk).
4. Empty-answer handling for hidden-set L4/L5 (unanswerable) questions.
