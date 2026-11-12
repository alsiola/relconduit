#!/usr/bin/env bash
# Reproduce the official RelBench LightGBM baseline (upstream examples/lightgbm_entity.py,
# vendored verbatim in experiments/rel-f1/baselines/upstream) on one task, for seeds 42-44.
# Usage: experiments/baselines/run_lightgbm.sh <dataset> <task>
# Each seed gets its own cache dir: upstream caches the materialised (sub-sampled) train set per
# task, so with a shared cache every seed after the first would reuse the first seed's sample.
set -euo pipefail
dataset=$1
task=$2
root=$(cd "$(dirname "$0")/../.." && pwd)
out="$root/runs/baselines/$dataset/$task"
mkdir -p "$out"
cd "$root/experiments/rel-f1/baselines/upstream"
for seed in 42 43 44; do
  start=$(date +%s)
  # OMP_NUM_THREADS=1: torch's and LightGBM's OpenMP runtimes collide on Apple Silicon.
  OMP_NUM_THREADS=1 uv run python lightgbm_entity.py --dataset "$dataset" --task "$task" \
    --seed "$seed" --cache_dir "$out/cache_s$seed" --pred_dir "$out/pred_s$seed" \
    > "$out/s$seed.log" 2>&1
  grep -E "^(Train|Val|Test):" "$out/s$seed.log" > "$out/s$seed.txt"
  echo "seconds: $(( $(date +%s) - start ))" >> "$out/s$seed.txt"
  cat "$out/s$seed.txt"
done
