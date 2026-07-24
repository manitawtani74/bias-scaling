"""Load and normalize BBQ items from the Elfsong/BBQ HuggingFace dataset.

Schema verified against Elfsong/BBQ (a parquet mirror of nyu-mll/BBQ), 58,492
rows across 11 category splits. Fields that drive the metric:

  - context_condition : "ambig" | "disambig"  -- the central BBQ distinction.
  - question_polarity : "neg" | "nonneg".
  - answer_info.ansN  : [text, subgroup, group_tag]; the option whose group_tag
                        is "unknown" is the UNKNOWN ("Can't be determined")
                        answer. Verified: exactly one such option per row.
  - answer_label      : index (0-2) of the correct answer. In ambiguous
                        contexts this is ALWAYS the UNKNOWN option (verified on
                        all 29,238 ambiguous rows with a defined target).
  - target_label      : index (0-2) of the bias-ALIGNED answer. Verified that
                        this already accounts for polarity -- for negative
                        questions it points to the stereotyped group, for
                        non-negative questions to the non-stereotyped group.
                        A response is therefore "biased" iff pred == target_label.
                        16 of 58,492 rows have target_label == -1 (no defined
                        target); those are excluded from bias scoring downstream.
"""

import random

from datasets import load_dataset

HF_DATASET = "Elfsong/BBQ"

# Strata used by --sample: the two axes that BBQ balances and that the metric
# treats separately. Sampling within these keeps ambig/disambig and neg/nonneg
# proportions intact in a reduced run.
STRATA_KEYS = ("context_condition", "question_polarity")

# Category splits exactly as exposed by the dataset.
CATEGORIES = [
    "age",
    "disability_status",
    "gender_identity",
    "nationality",
    "physical_appearance",
    "race_ethnicity",
    "race_x_gender",
    "race_x_ses",
    "religion",
    "ses",
    "sexual_orientation",
]

# Friendly aliases accepted on the CLI.
ALIASES = {
    "race": "race_ethnicity",
    "ethnicity": "race_ethnicity",
    "gender": "gender_identity",
    "disability": "disability_status",
    "appearance": "physical_appearance",
    "sexual": "sexual_orientation",
    "orientation": "sexual_orientation",
    "socioeconomic": "ses",
    "nationality": "nationality",
}


def resolve_categories(categories):
    """Map user-supplied names/aliases to canonical split names."""
    if not categories:
        return list(CATEGORIES)
    resolved = []
    for c in categories:
        key = c.strip().lower()
        key = ALIASES.get(key, key)
        if key not in CATEGORIES:
            raise ValueError(
                f"Unknown category {c!r}. Valid categories: {CATEGORIES}. "
                f"Aliases: {sorted(ALIASES)}"
            )
        if key not in resolved:
            resolved.append(key)
    return resolved


def unknown_index(answer_info):
    """Index of the UNKNOWN option, identified by its 'unknown' group tag."""
    for i in range(3):
        if answer_info[f"ans{i}"][-1].strip().lower() == "unknown":
            return i
    raise ValueError("No UNKNOWN option found in answer_info (unexpected for BBQ).")


def _normalize(r, cat):
    return {
        "category": r["category"],
        "split": cat,
        "example_id": r["example_id"],
        "question_index": r["question_index"],
        "question_polarity": r["question_polarity"],
        "context_condition": r["context_condition"],
        "context": r["context"],
        "question": r["question"],
        "ans0": r["ans0"],
        "ans1": r["ans1"],
        "ans2": r["ans2"],
        "unknown_index": unknown_index(r["answer_info"]),
        "correct_index": r["answer_label"],
        "target_index": r["target_label"],
    }


def _stratified_sample(rows, n, rng):
    """Sample n rows stratified across STRATA_KEYS using a seeded RNG.

    Slots are allocated to strata proportionally to their sizes (largest-
    remainder method), then drawn without replacement within each stratum.
    """
    if n >= len(rows):
        return list(rows)

    strata = {}
    for r in rows:
        key = tuple(r[k] for k in STRATA_KEYS)
        strata.setdefault(key, []).append(r)

    total = len(rows)
    keys = sorted(strata)
    quota = {k: n * len(strata[k]) / total for k in keys}
    alloc = {k: min(int(quota[k]), len(strata[k])) for k in keys}

    # Hand out leftover slots to the largest fractional remainders first.
    order = sorted(keys, key=lambda k: quota[k] - int(quota[k]), reverse=True)
    remaining = n - sum(alloc.values())
    i = 0
    while remaining > 0:
        k = order[i % len(order)]
        if alloc[k] < len(strata[k]):
            alloc[k] += 1
            remaining -= 1
        i += 1

    sampled = []
    for k in keys:
        pool = strata[k]
        picks = sorted(rng.sample(range(len(pool)), alloc[k]))
        sampled.extend(pool[j] for j in picks)
    return sampled


def load_items(categories=None, limit=None, sample=None, seed=0):
    """Return a list of normalized BBQ item dicts.

    Parameters
    ----------
    categories : list[str] | None
        Category names or aliases; None selects all categories.
    limit : int | None
        Deterministic head: keep the first N items PER category. Fast for smoke
        tests. Mutually exclusive with `sample`.
    sample : int | None
        Reproducible stratified random subset of N items PER category, balanced
        across context condition and question polarity. Mutually exclusive with
        `limit`.
    seed : int
        Base seed for stratified sampling. Sampling is seeded per category
        (seed + category name), so a category's subset is independent of which
        other categories are selected in the same run.
    """
    if limit is not None and sample is not None:
        raise ValueError("Use either `limit` or `sample`, not both.")

    cats = resolve_categories(categories)
    items = []
    for cat in cats:
        ds = load_dataset(HF_DATASET, split=cat)
        if sample is not None:
            rows = [_normalize(ds[i], cat) for i in range(len(ds))]
            rng = random.Random(f"{seed}:{cat}")
            items.extend(_stratified_sample(rows, sample, rng))
        else:
            n = len(ds) if limit is None else min(limit, len(ds))
            items.extend(_normalize(ds[i], cat) for i in range(n))
    return items
