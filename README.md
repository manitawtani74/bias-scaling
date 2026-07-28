# bias-scaling — does social bias scale with model size?

**On BBQ, larger Qwen2.5 models are simultaneously more accurate and more stereotypically biased: capability and bias rise together.**

## TL;DR

I evaluated the Qwen2.5 family (0.5B, 1.5B, 3B, 7B) on the BBQ bias benchmark, scoring answer options by log-likelihood rather than free-text generation. As size grows, the models get much better at reading contexts that actually contain the answer (disambiguated accuracy climbs from 0.57 to 0.94). But in ambiguous contexts — where the right answer is "unknown" and any confident guess is unwarranted — the bigger models lean *harder* on the social stereotype, with the ambiguous-context bias score rising from 0.08 at 0.5B to roughly 0.25–0.31 at the larger sizes. So within one family, scale buys capability and stereotyping at the same time. These are preliminary single-seed numbers on a 200-item-per-category sample.

![Bias and accuracy vs. Qwen2.5 size](results/scaling_letterperm.png)

## Background

BBQ (the [Bias Benchmark for QA](https://github.com/nyu-mll/BBQ)) presents each question twice. In the **ambiguous** version the context gives no basis for an answer, so the correct response is always "unknown" and any committed choice of a social group exposes a prior. In the **disambiguated** version the context names who did what, so there is a factually correct answer and bias shows up only as skewed errors. The bias score runs from −1 (systematically anti-stereotypical) through 0 (unbiased) to +1 (systematically stereotypical). Ambiguous contexts are the more revealing of the two, since a model has to actively guess to be scored as biased.

## Method

Each item's three answer options are ranked by the log-likelihood the model assigns them, and the top option is taken as the prediction — no generation, no parsing. The headline results use **letter mode**: options are presented as A/B/C and the letter token is scored, with the score averaged over all six option orderings so that any fixed A/B/C position preference cancels out. I also run a plainer **text mode** that scores the answer text directly. Every configuration uses a reproducible, stratified sample of 200 items per category (balanced across ambiguous/disambiguated and question polarity) at seed 0, run on GPU. Bias and accuracy are computed per category and overall following the standard BBQ definitions in `src/metrics.py`.

## Results

Overall letter+permute scores across the four sizes (200 items/category, seed 0):

| size | accuracy (disambiguated) | bias (ambiguous) | bias (disambiguated) |
|------|-------------------------:|-----------------:|---------------------:|
| 0.5B | 0.574 | 0.082 | 0.098 |
| 1.5B | 0.845 | 0.251 | 0.066 |
| 3B   | 0.903 | 0.305 | 0.010 |
| 7B   | 0.941 | 0.237 | 0.030 |

Disambiguated accuracy rises monotonically — larger models simply read the evidence better — while disambiguated bias stays near zero and even shrinks, because a model that gets the answer right has little room to err in a stereotyped direction. The action is in the ambiguous column: bias climbs steeply from 0.5B through 3B and remains high at 7B, meaning the more capable models are precisely the ones that fill an evidential vacuum with the stereotype. Text-mode scoring shows the same shape over the sizes measured so far (the 7B text point is still backfilling; the figure fills in automatically once its CSV lands, via `python scripts/plot_scaling.py`).

![Text-mode scaling](results/scaling_text.png)

## A note on the MPS anomaly

An earlier round of these numbers was computed on Apple's MPS backend and then thrown out. While validating the scoring I found MPS returning logits that diverged from CPU by several nats on a subset of inputs — enough to flip predictions — including a causal-masking violation where an appended token changed the logits at an earlier position, which a correct causal model cannot do. The failure is intermittent across process runs, so it resisted a clean deterministic repro. `repro_mps_bug.py` probes it (causal-masking invariant plus MPS-vs-CPU agreement across a length sweep, repeated); everything reported here was computed on GPU/CPU, never MPS.

## Limitations

This is one seed, one sample size, and one model family, so treat small differences as noise and the whole thing as a pilot rather than a measurement — the ambiguous-context bias is already non-monotonic at the top end (it peaks at 3B and dips at 7B), which more seeds would either confirm or wash out. The two largest models run in bfloat16 to fit a 16 GB GPU while the two smallest run in float32, so precision is not held constant across the curve. Whether the capability-and-bias-together pattern generalizes beyond Qwen2.5 is untested.

## Reproduce

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# One configuration locally (use --device cpu; MPS is not trusted — see above)
python -m src.evaluate --model Qwen/Qwen2.5-0.5B --device cpu \
    --scoring letter --permute --sample 200 --seed 0

python scripts/plot_scaling.py     # rebuild the figures from the metrics CSVs
```

The full four-size sweep is impractical on a laptop; `notebooks/kaggle_sweep.ipynb` runs it on a free Kaggle GPU (set Accelerator = GPU and Internet = On, then Run All), and `notebooks/kaggle_seeds.ipynb` adds seeds for error bars. Key flags: `--device {cpu,cuda}`, `--scoring {text,letter}`, `--permute`, `--sample N`, `--seed`, `--dtype`. Layout: `src/` holds the pipeline (`data.py`, `scoring.py`, `metrics.py`, `evaluate.py`), `scripts/plot_scaling.py` builds the figures, `notebooks/` holds the GPU sweeps, and `results/` holds the committed CSVs and plots.
