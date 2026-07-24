# bias-scaling

**Does social bias vary systematically with model size within a single model
family, measured on BBQ?**

That is the only question this repo is trying to answer. Everything else is
method and plumbing.

## Method

- **Benchmark:** [BBQ](https://github.com/nyu-mll/BBQ) (Bias Benchmark for QA),
  via [`Elfsong/BBQ`](https://huggingface.co/datasets/Elfsong/BBQ) — 11 social
  categories, each item in an **ambiguous** and a **disambiguated** context
  (the distinction the bias metric turns on).
- **Scoring:** no free-text generation. Each answer option is scored by
  **log-likelihood** and the highest-scoring option is the prediction. Two modes:
  - `text` — score the answer text as a continuation (classic BBQ cloze).
  - `letter` — present options as A/B/C and score the letter token; optional
    `--permute` averages over all 6 option orderings to cancel letter-position bias.
- **Metric:** standard BBQ bias scores for ambiguous and disambiguated contexts,
  plus accuracy, reported per category (`src/metrics.py`).
- **Model family:** **Qwen2.5** (ungated), size points **0.5B / 1.5B / 3B / 7B**.

## Status — in progress, no results claimed

⚠️ **All numbers computed earlier on the Apple MPS backend have been discarded.**
While validating letter mode I observed the MPS backend returning **wrong logits**
for a subset of inputs (divergences of 5–13 nats vs CPU, enough to flip the
predicted answer). Every prior MPS run is therefore untrustworthy and no bias
results are being reported yet.

The evaluation is being **re-run on CPU** (`--device cpu`), which matched the
reference on every input tested. This is the only compute path currently trusted.
Until those runs complete and are checked, **treat this repo as method + tooling,
not findings.**

## Open question: the MPS logit anomaly

The MPS divergence is real but **has not been reproduced on demand.** It appeared
strongly and repeatedly in some sessions (including a causal-masking violation —
an appended token changing the logits at an *earlier* position, which is
impossible for a correct causal model) yet was **completely absent in others on
identical inputs.** It is deterministic within a process but varies across runs,
suggesting an MPS kernel/allocator-state or load/timing dependence rather than a
clean function of input shape.

`repro_mps_bug.py` probes this: it checks the causal-masking invariant and
MPS-vs-CPU agreement across a sweep of sequence lengths, repeated (since the
failure is intermittent). Observed on: Apple Silicon (arm64), macOS 14.3,
Python 3.12.4, torch 2.13.0, transformers 5.14.1.

```bash
python repro_mps_bug.py --repeats 10     # exits 1 if it catches a divergence
```

If you can reproduce it deterministically, or it's a known PyTorch-MPS issue,
that would resolve whether MPS is usable here at all.

## Reproduction

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Smoke test (20 items, one category)
python -m src.evaluate --device cpu --category age --limit 20

# Trusted evaluation: reproducible stratified 200/category, both modes, on CPU
python -m src.evaluate --model Qwen/Qwen2.5-0.5B --device cpu --scoring text   --sample 200 --seed 0
python -m src.evaluate --model Qwen/Qwen2.5-0.5B --device cpu --scoring letter --permute --sample 200 --seed 0
# ...repeat for Qwen/Qwen2.5-1.5B, -3B, -7B (use --dtype bfloat16 for 7B)
```

Each run writes a raw per-item CSV (every option's log-likelihood, the
prediction, and correctness/bias flags — enough to re-derive metrics without
rerunning) plus a `*_metrics.csv` summary to `results/`.

Key flags: `--device {cpu,mps,auto}` (use **cpu**), `--scoring {text,letter}`,
`--permute`, `--sample N` / `--limit N`, `--seed`, `--dtype`.

## Layout

```
src/
  data.py       load + normalize BBQ (schema notes inline)
  scoring.py    log-likelihood scoring, text + letter modes
  metrics.py    BBQ bias-score computation
  evaluate.py   CLI entry point
repro_mps_bug.py  MPS logit-divergence probe
results/        per-run CSV outputs (committed; the actual output)
```
