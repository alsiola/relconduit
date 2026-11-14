#!/usr/bin/env bash
# Issue 29's phase-2 cloud run: embed with both encoders side by side, then the val ablations.
# Checkpointed folder: runs/p29 (embeddings, fits, scores); pair tables in runs/p29-pairs.
set -u
D=experiments/rel-stack/post-post-related/dense-ranker
mkdir -p runs/p29/logs
echo "[cloud29] $(date -u +%H:%M:%S) embedding start"
OMP_NUM_THREADS=12 uv run python -W ignore $D/embed.py --model sentence-transformers/all-MiniLM-L6-v2 \
  2>&1 | grep --line-buffered -v "Loading weights\|Batches" | sed -u 's/^/[minilm] /' &
OMP_NUM_THREADS=12 uv run python -W ignore $D/embed.py --model BAAI/bge-small-en-v1.5 \
  2>&1 | grep --line-buffered -v "Loading weights\|Batches" | sed -u 's/^/[bge] /' &
wait
ls -la runs/p29/emb
echo "[cloud29] $(date -u +%H:%M:%S) ablations start"
OMP_NUM_THREADS=24 uv run python -W ignore $D/run.py "$@" 2>&1 | sed -u 's/^/[ablate] /'
rc=${PIPESTATUS[0]}
echo "[cloud29] $(date -u +%H:%M:%S) done rc=$rc"
exit $rc
