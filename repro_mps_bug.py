"""Minimal reproduction: PyTorch MPS backend returns wrong logits for some inputs.

FINDING
-------
Running Qwen2.5 (0.5B / 1.5B) on the Apple-Silicon **MPS** backend sometimes
produces logits that are wrong by 5-13 nats -- large enough to flip the argmax
of a next-token prediction. This is NOT ordinary float rounding noise (which is
~1e-4 here). CPU is always correct; MPS diverges on a subset of inputs.

Two independent symptoms were observed, both checked by this script:

  1. CAUSALITY VIOLATION (self-contained, needs no CPU reference).
     In a causal LM the logits at position i depend only on tokens 0..i, so the
     next-token distribution after a prompt must be identical whether you run
     the model on the prompt alone or on the prompt PLUS one appended token
     (reading the same position i = len(prompt)-1). On MPS this invariant is
     sometimes violated -- the appended token leaks into an earlier position --
     which is unambiguously a backend bug. This is exactly what corrupts the
     per-token log-likelihood scoring in src/scoring.py, which runs the model
     over prompt+continuation and reads earlier positions.

  2. MPS-vs-CPU DIVERGENCE on the plain next-token logits for the same input.

IMPORTANT: THE BUG IS INTERMITTENT.
It reproduced strongly and repeatedly in some sessions (with a seemingly clean
affected sequence-length band, e.g. ~57-63 tokens) yet was completely absent in
others on the identical inputs -- so it is not a clean deterministic function of
sequence length. It appears to depend on MPS kernel/allocator state and possibly
system load or timing. Because of this, the script sweeps many lengths and
repeats the sweep several times, reporting the worst divergence seen; you may
need several runs, or a machine under different load, to catch it. Within a
single process it is deterministic (re-running the same tensor reproduces the
same result), and `attn_implementation` ("sdpa" vs "eager") makes no difference.

Impact for this repo: a BBQ evaluation on MPS silently corrupts a subset of
items. Run evaluations with `--device cpu` (see src/evaluate.py) when
correctness matters; CPU matched the reference on every input tested.

AFFECTED ENVIRONMENT (where this was observed)
----------------------------------------------
  - Hardware     : Apple Silicon (arm64), M-series
  - macOS        : 14.3
  - Python       : 3.12.4
  - torch        : 2.13.0
  - transformers : 5.14.1
  - Model        : Qwen/Qwen2.5-0.5B (default; --model to change)

Other torch / macOS versions may shift, hide, or worsen the effect.

HOW TO RUN
----------
    source venv/bin/activate
    python repro_mps_bug.py                         # sweep lengths x repeats
    python repro_mps_bug.py --model Qwen/Qwen2.5-1.5B
    python repro_mps_bug.py --repeats 10 --threshold 0.01

Exit code is 1 if any divergence above --threshold is seen (0 otherwise), so the
script doubles as a (flaky) regression check.
"""

import argparse
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# A diverse natural-language passage. Real, varied tokens matter: cycling a few
# tokens to pad length did NOT trigger the bug, whereas varied content did.
# Length is swept by taking the last L tokens of this passage.
PASSAGE = (
    "The committee reviewed the quarterly financial report on Tuesday afternoon, "
    "noting that revenue from international markets had grown substantially while "
    "domestic expenditures remained flat despite inflationary pressure across "
    "several key supply categories including electronics, textiles, and machinery. "
    "Analysts recommended a cautious approach to expansion, emphasizing liquidity "
    "and debt reduction before committing to any acquisition of the smaller "
    "competing regional firms that had recently entered the northern market. "
    "A grandmother and her grandson were also mentioned, somewhat incongruously, "
    "in a footnote about consumer technology adoption among different age groups."
)


def next_token_logprobs(model, ids, position, device):
    """Log-probs over the vocab predicted at `position` of `ids`."""
    with torch.no_grad():
        logits = model(ids.to(device)).logits[0, position, :].float()
    return torch.log_softmax(logits, dim=-1).cpu()


def check_length(model, all_ids, filler_id, length, device):
    """Return (causality_gap, mps_vs_cpu_gap) for a length-`length` prompt.

    causality_gap: max|logits(prompt) - logits(prompt+token) at the same
                   final position| on `device`. Must be ~0 for a causal model.
    mps_vs_cpu_gap: max|logits on device - logits on cpu| for the prompt.
    """
    prompt = torch.tensor([all_ids[-length:]])
    extended = torch.tensor([all_ids[-length:] + [filler_id]])
    last = length - 1

    lp_prompt_dev = next_token_logprobs(model, prompt, last, device)
    lp_extended_dev = next_token_logprobs(model, extended, last, device)
    causality_gap = (lp_prompt_dev - lp_extended_dev).abs().max().item()

    model.to("cpu")
    lp_prompt_cpu = next_token_logprobs(model, prompt, last, torch.device("cpu"))
    model.to(device)
    cross_gap = (lp_prompt_dev - lp_prompt_cpu).abs().max().item()

    return causality_gap, cross_gap


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--min-len", type=int, default=45)
    ap.add_argument("--max-len", type=int, default=95)
    ap.add_argument("--repeats", type=int, default=3,
                    help="Repeat the whole sweep this many times (the bug is intermittent).")
    ap.add_argument("--threshold", type=float, default=0.01,
                    help="Flag a divergence when a gap exceeds this (nats).")
    args = ap.parse_args(argv)

    device = torch.device("mps")
    if not torch.backends.mps.is_available():
        print("MPS is not available on this machine; nothing to reproduce.")
        return 0

    print(f"Loading {args.model} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).eval()
    all_ids = tokenizer(PASSAGE).input_ids
    filler_id = tokenizer(" the", add_special_tokens=False).input_ids[0]
    max_len = min(args.max_len, len(all_ids) - 1)

    worst_causality = 0.0
    worst_cross = 0.0
    hits = []
    for rep in range(1, args.repeats + 1):
        for length in range(args.min_len, max_len + 1):
            model.to(device)
            causality_gap, cross_gap = check_length(model, all_ids, filler_id, length, device)
            worst_causality = max(worst_causality, causality_gap)
            worst_cross = max(worst_cross, cross_gap)
            if max(causality_gap, cross_gap) > args.threshold:
                hits.append((rep, length, causality_gap, cross_gap))
                print(f"  rep {rep} len {length}: causality_gap={causality_gap:.4f} "
                      f"mps_vs_cpu={cross_gap:.4f}   <-- MPS WRONG")
        print(f"repeat {rep}/{args.repeats} done: "
              f"worst causality={worst_causality:.5f} worst mps_vs_cpu={worst_cross:.5f}",
              flush=True)

    print("-" * 64)
    print(f"worst causality-violation gap : {worst_causality:.5f} nats  (should be ~0)")
    print(f"worst MPS-vs-CPU gap          : {worst_cross:.5f} nats  (should be ~1e-4)")
    if hits:
        print(f"REPRODUCED: {len(hits)} (repeat,length) inputs exceeded "
              f"threshold {args.threshold}.")
        return 1
    print("Not reproduced this run. The bug is intermittent -- try again, raise "
          "--repeats, or run under different system load. CPU remained correct.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
