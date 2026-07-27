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

## Results

Preliminary, single-seed results from the **letter + permutation** configuration
(letter-position bias removed via 6-way answer-order averaging), run on GPU
(CUDA) through the Kaggle notebook. Overall scores across the four Qwen2.5 sizes,
200 items/category, seed 0:

| size | accuracy (disambig) | bias (ambiguous) | bias (disambig) |
|------|--------------------:|-----------------:|----------------:|
| 0.5B | 0.574 | 0.082 | 0.098 |
| 1.5B | 0.845 | 0.251 | 0.066 |
| 3B   | 0.903 | 0.305 | 0.010 |
| 7B   | 0.941 | 0.237 | 0.030 |

**Main finding — capability and stereotypical bias rise together.**
Disambiguated-context accuracy climbs monotonically with size
(0.57 → 0.85 → 0.90 → 0.94): larger models read the disambiguating evidence
better. But in **ambiguous** contexts — where the correct answer is always
"unknown" and any committed guess reveals a prior — the stereotypical bias score
*also* rises with size, from 0.08 at 0.5B to 0.25–0.31 at 1.5B–3B (0.24 at 7B).
So on BBQ the more capable Qwen2.5 models are the ones that lean harder on the
social stereotype when forced to guess. Disambiguated-context bias stays small
and even falls with size — there the big models mostly just get the answer right.

![letter+permute scaling](results/scaling_letterperm.png)

Text-mode scoring (plain answer text, no letter format) shows the same overall
shape at the sizes available so far; the 7B text point is pending (backfilling
separately) and the figure fills it in automatically once the CSV lands.

![text scaling](results/scaling_text.png)

Regenerate both figures from the metrics CSVs with `python scripts/plot_scaling.py`.

### Limitations

- **Single seed (0), one sample of 200 items/category** — no error bars yet;
  treat small differences as noise.
- **One model family** (Qwen2.5); whether the trend generalizes is untested.
- **Mixed precision:** 0.5B/1.5B in float32, 3B/7B in **bfloat16** (to fit a
  16 GB GPU), so the two largest points differ in numerical precision from the
  two smallest.
- **7B text run pending**, so the text-mode figure currently has three points.
- Ambiguous-context bias is **non-monotonic at the top end** (peaks at 3B, dips
  slightly at 7B) — more seeds and sizes are needed to know whether that is real.
- **Apple MPS results are excluded entirely** — see the open question below.

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

Key flags: `--device {cpu,mps,auto,cuda}` (use **cpu** on the laptop, **cuda** on
a GPU), `--scoring {text,letter}`, `--permute`, `--sample N` / `--limit N`,
`--seed`, `--dtype`.

### Run the sweep on Kaggle (GPU)

The full four-size sweep is slow on laptop CPU (the MPS backend is not trusted —
see above). `notebooks/kaggle_sweep.ipynb` runs it on a free Kaggle GPU instead:
it clones this repo, installs deps, and runs all four Qwen2.5 sizes
(0.5B/1.5B/3B/7B) in both `text` and `letter --permute` modes with `--device cuda`,
writing CSVs to `/kaggle/working/results/` (saved as notebook output).

1. On [kaggle.com](https://www.kaggle.com/code) → **New Notebook** → *File →
   Import Notebook* and upload `notebooks/kaggle_sweep.ipynb`.
2. In the right-hand **Settings** panel set **Accelerator = GPU** (a single 16 GB
   T4 or P100 is enough) and **Internet = On**.
3. Optional: *Add-ons → Secrets* → add a secret named **`HF_TOKEN`** with a
   Hugging Face token (avoids download rate limits; Qwen2.5 is ungated so it is
   not required).
4. **Run All**. Download the resulting CSVs from the notebook's **Output** tab
   (or *Save Version* to persist them), then drop them into `results/` locally.

`0.5B/1.5B/3B` run in float32; **7B uses bfloat16** to fit in 16 GB (editable in
the sweep cell — switch 3B to bfloat16 too if it OOMs).

## Layout

```
src/
  data.py       load + normalize BBQ (schema notes inline)
  scoring.py    log-likelihood scoring, text + letter modes
  metrics.py    BBQ bias-score computation
  evaluate.py   CLI entry point
scripts/
  plot_scaling.py   build the scaling figures from the metrics CSVs
notebooks/          Kaggle GPU sweep + 7B-text backfill
repro_mps_bug.py    MPS logit-divergence probe
results/            per-run CSV outputs + scaling_*.png (committed; the actual output)
```
