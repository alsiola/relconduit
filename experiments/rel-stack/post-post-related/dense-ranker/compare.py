"""Paired bootstraps between the top val variants of run 1, from the saved top-100 tables.

Reads ``runs/p29/ablate/scores/<slug>_val_top100.npy`` (one per variant and seed), takes the
per-row AP@100 averaged over the 5 ranker seeds, and compares variants row by row (2,000
resamples of the 226 val rows). Val only; test is never read.

    uv run python experiments/rel-stack/post-post-related/dense-ranker/compare.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import relbench

sys.path.insert(0, str(Path(__file__).parent))
from run import SEEDS, ap_rows, paired_bootstrap

from relconduit.link_lgbm import _slug

SCORES = Path("runs/p29/ablate/scores")
LEADER = "dense_both_300+extra+num_leaves=31"
PAIRS = [
    (LEADER, "ref"),
    (LEADER, "dense_minilm_100"),
    (LEADER, "dense_minilm_300"),
    (LEADER, "dense_both_300"),
    (LEADER, "dense_both_300+extra"),
    (LEADER, "dense_both_300+extra+min_child_samples=300"),
    ("dense_both_300+extra", "dense_both_300"),
    ("dense_both_300", "dense_minilm_300"),
    ("dense_both_300", "dense_minilm_100"),
    ("dense_minilm_300", "dense_minilm_100"),
]


def main():
    task = relbench.load_dataset("stanford-star/relbench-v1/rel-stack").load_task(
        "post-post-related"
    )
    val = task.get_table("val", mask_input_cols=False).df.reset_index(drop=True)
    true = [{int(d) for d in v} for v in val[task.dst_entity_col]]

    def ap(tag: str) -> np.ndarray:
        keys = [json.dumps({"variant": tag, "seed": s}, sort_keys=True) for s in SEEDS]
        tables = [np.load(SCORES / f"{_slug(k)}_val_top{task.eval_k}.npy") for k in keys]
        return np.mean([ap_rows(t, true) for t in tables], axis=0)

    cache: dict[str, np.ndarray] = {}
    for a, b in PAIRS:
        for t in (a, b):
            if t not in cache:
                cache[t] = ap(t)
        r = paired_bootstrap(cache[a], cache[b])
        print(
            f"{a} vs {b}: delta {r['delta']:+.4f}, 95% CI [{r['ci95'][0]:+.4f}, "
            f"{r['ci95'][1]:+.4f}], P(delta <= 0) {r['p_le_0']:.3f}"
        )


if __name__ == "__main__":
    main()
