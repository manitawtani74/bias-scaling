"""CLI entry point: evaluate a causal LM on BBQ via log-likelihood scoring.

Example
-------
    python -m src.evaluate --limit 20
    python -m src.evaluate --model Qwen/Qwen2.5-3B --sample 200 --category race gender
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import load_items
from src.device import get_device
from src.metrics import compute_metrics
from src.scoring import predict

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}

# Qwen2.5 is ungated and ships in 0.5B / 1.5B / 3B / 7B, giving four size points
# within one family for the scaling study.
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Evaluate a causal LM on BBQ.")
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"HuggingFace causal LM name or path. Default: {DEFAULT_MODEL}",
    )
    p.add_argument(
        "--category",
        nargs="+",
        default=None,
        help="One or more BBQ categories/aliases (e.g. race gender age). Default: all.",
    )
    sub = p.add_mutually_exclusive_group()
    sub.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Deterministic head: first N items PER category (fast smoke test).",
    )
    sub.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Reproducible stratified random subset of N items PER category "
        "(balanced across context condition and polarity).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed for --sample. Default: 0.",
    )
    p.add_argument(
        "--scoring",
        choices=["text", "letter"],
        default="text",
        help="Scoring mode: 'text' scores the answer text as a continuation; "
        "'letter' presents A/B/C options and scores the letter token (removes "
        "the fluency penalty against abstention). Default: text.",
    )
    p.add_argument(
        "--permute",
        action="store_true",
        help="Letter mode only: average each option's score over all 6 "
        "letter->option assignments to cancel letter position bias.",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Path for the raw per-item CSV. Default: results/<model>_<timestamp>.csv",
    )
    p.add_argument(
        "--dtype",
        choices=sorted(DTYPES),
        default="float32",
        help="Model dtype. float32 is the safest default on Apple MPS.",
    )
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help="Compute device. 'auto' picks MPS/CUDA/CPU. NOTE: the MPS backend "
        "returns wrong logits for some sequence shapes with these models, so use "
        "'cpu' when correctness matters.",
    )
    args = p.parse_args(argv)
    if args.permute and args.scoring != "letter":
        p.error("--permute is only valid with --scoring letter.")
    return args


def load_model(model_name, dtype, device):
    print(f"Loading tokenizer and model: {model_name} ({dtype} on {device}) ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=DTYPES[dtype])
    model.to(device)
    model.eval()
    return tokenizer, model


def run(items, tokenizer, model, device, scoring="text", permute=False):
    rows = []
    total = len(items)
    for idx, item in enumerate(items, 1):
        scores, pred_index = predict(
            model, tokenizer, item, device, scoring=scoring, permute=permute
        )
        pred_is_unknown = pred_index == item["unknown_index"]
        is_correct = pred_index == item["correct_index"]
        has_target = item["target_index"] >= 0
        pred_is_biased = (
            bool(has_target and not pred_is_unknown and pred_index == item["target_index"])
            if has_target
            else None
        )
        rows.append(
            {
                **{k: item[k] for k in (
                    "category", "split", "example_id", "question_index",
                    "question_polarity", "context_condition", "context", "question",
                    "ans0", "ans1", "ans2", "unknown_index", "correct_index", "target_index",
                )},
                "ll0_sum": scores[0]["sum_logprob"],
                "ll1_sum": scores[1]["sum_logprob"],
                "ll2_sum": scores[2]["sum_logprob"],
                "ll0_mean": scores[0]["mean_logprob"],
                "ll1_mean": scores[1]["mean_logprob"],
                "ll2_mean": scores[2]["mean_logprob"],
                "pred_index": pred_index,
                "pred_is_unknown": pred_is_unknown,
                "is_correct": is_correct,
                "pred_is_biased": pred_is_biased,
                "scoring": scoring,
                "permute": permute,
            }
        )
        if idx % 10 == 0 or idx == total:
            print(f"  scored {idx}/{total} items", flush=True)
    return pd.DataFrame(rows)


def main(argv=None):
    args = parse_args(argv)
    device = get_device() if args.device == "auto" else torch.device(args.device)

    print("Loading BBQ items ...", flush=True)
    items = load_items(
        categories=args.category, limit=args.limit, sample=args.sample, seed=args.seed
    )
    print(f"Loaded {len(items)} items across categories: "
          f"{sorted({it['split'] for it in items})}", flush=True)

    tokenizer, model = load_model(args.model, args.dtype, device)

    start = time.time()
    df = run(items, tokenizer, model, device, scoring=args.scoring, permute=args.permute)
    print(f"Scoring finished in {time.time() - start:.1f}s", flush=True)

    # Resolve output paths.
    safe_model = args.model.replace("/", "__")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_csv = Path(args.output) if args.output else Path("results") / f"{safe_model}_{stamp}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote raw per-item results -> {out_csv}", flush=True)

    metrics = compute_metrics(df)
    summary_csv = out_csv.with_name(out_csv.stem + "_metrics.csv")
    metrics.to_csv(summary_csv, index=False)
    print(f"Wrote metrics summary   -> {summary_csv}", flush=True)

    print("\n=== BBQ metrics ===")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(metrics.to_string(index=False))

    meta = {
        "model": args.model,
        "dtype": args.dtype,
        "device": str(device),
        "categories": sorted({it["split"] for it in items}),
        "limit": args.limit,
        "sample": args.sample,
        "seed": args.seed,
        "scoring": args.scoring,
        "permute": args.permute,
        "n_items": len(df),
        "raw_csv": str(out_csv),
        "metrics_csv": str(summary_csv),
    }
    print("\n" + json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
