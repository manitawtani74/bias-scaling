#!/usr/bin/env bash
#
# Run the complete bias-scaling study on a single dedicated GPU (e.g. an A100
# with ample VRAM):
#
#     4 Qwen2.5 sizes (0.5B/1.5B/3B/7B) × seeds 0,1,2 × 2 modes (text, letter+permute)
#     = 24 evaluations, float32 throughout.
#
# Outputs go straight to results/ with seed-tagged names, e.g.
#     qwen7b_letterperm_cuda_seed2.csv   (+ _metrics.csv)
#
# Resumable: any run whose output CSV already exists is skipped, so a disconnect
# just picks up where it left off (evaluate.py writes the CSV only after a run
# finishes, so a present file always means a complete run). Prints per-run
# wall-clock and a total at the end.
#
# Usage:  bash scripts/run_full_sweep.sh

set -uo pipefail

cd "$(dirname "$0")/.."  # repo root

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Activate the project venv if present.
if [ -f venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source venv/bin/activate
fi

mkdir -p results

SIZES=("0.5B" "1.5B" "3B" "7B")
SEEDS=(0 1 2)
MODES=("text" "letterperm")

tag_of() {
    case "$1" in
        0.5B) echo "05b" ;;
        1.5B) echo "15b" ;;
        3B)   echo "3b"  ;;
        7B)   echo "7b"  ;;
    esac
}

n_done=0
n_skip=0
n_fail=0
total_start=$SECONDS

for seed in "${SEEDS[@]}"; do
    for size in "${SIZES[@]}"; do
        tag="$(tag_of "$size")"
        model="Qwen/Qwen2.5-${size}"
        for mode in "${MODES[@]}"; do
            base="qwen${tag}_${mode}_cuda_seed${seed}"
            out="results/${base}.csv"

            if [ -f "$out" ]; then
                printf 'SKIP  %-34s (output exists)\n' "$base"
                n_skip=$((n_skip + 1))
                continue
            fi

            # text = default scoring; letterperm = letter mode + 6-way permutation.
            extra=()
            if [ "$mode" = "letterperm" ]; then
                extra=(--scoring letter --permute)
            fi

            printf 'RUN   %s ...\n' "$base"
            start=$SECONDS
            python -m src.evaluate \
                --model "$model" --device cuda --dtype float32 \
                --sample 200 --seed "$seed" "${extra[@]}" --output "$out"
            rc=$?
            elapsed=$((SECONDS - start))

            if [ $rc -eq 0 ] && [ -f "$out" ]; then
                printf 'DONE  %-34s %5ds\n' "$base" "$elapsed"
                n_done=$((n_done + 1))
            else
                printf '!! FAIL %-32s exit=%d after %ds; leaving no CSV so it retries next run\n' \
                    "$base" "$rc" "$elapsed"
                n_fail=$((n_fail + 1))
            fi
        done
    done
done

total=$((SECONDS - total_start))
printf '\n=== sweep finished: %d run, %d skipped, %d failed (of 24) — wall-clock %ds (%dm%02ds) ===\n' \
    "$n_done" "$n_skip" "$n_fail" "$total" "$((total / 60))" "$((total % 60))"

# Non-zero exit if anything failed, so a CI/cron wrapper can notice.
[ "$n_fail" -eq 0 ]
