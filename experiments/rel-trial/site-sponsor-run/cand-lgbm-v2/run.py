"""rel-trial/site-sponsor-run, issue 28: the train-size curve and the candidate ablation. See
README.md.

Plans (``--plan``), each a :class:`relconduit.link_search.SearchPlan` on the baseline's
candidates and features (``../cand-lgbm-ranker/run.py``):

* ``curve``: the baseline candidate set, ranker config fixed at the baseline's choice, fitted on
  25k, 50k, 100k, 200k and all 669,310 train seed rows (nested subsets), 3 seeds each.
* ``ablation``: the baseline plus each new source of :data:`relconduit.trial_edges.NEW_PATHS`
  on its own, and all of them together, each at 25k train seed rows, 3 seeds. Every candidate
  set gets its own real-database pair leakage check before any fit.

Every other option of :func:`relconduit.link_search.main` passes through (``--smoke``,
``--work-dir``, ``--pairs-dir``, ``--test``).

    uv run python experiments/rel-trial/site-sponsor-run/cand-lgbm-v2/run.py --plan ablation --smoke
"""

import argparse
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path as _P

from relconduit.link_search import SearchPlan, main
from relconduit.trial_edges import EDGES, NEW_PATHS

_spec = importlib.util.spec_from_file_location(
    "base_run", _P(__file__).parents[1] / "cand-lgbm-ranker" / "run.py"
)
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

# The baseline's chosen config (issue 12), the reference for every comparison here.
REF = {
    "objective": "lambdarank",
    "num_leaves": 63,
    "min_child_samples": 20,
    "feature_fraction": 0.7,
}


def plus(*names: str, **changes):
    """The baseline candidate set with the named new sources added."""
    return replace(
        base.LINK,
        edges=EDGES,
        paths={**base.LINK.paths, **{n: NEW_PATHS[n] for n in names}},
        **changes,
    )


VARIANTS = {
    "base": base.LINK,
    "cosponsor": plus("cosponsor"),
    "condition": plus("condition"),
    "geo": plus("zip", "state"),
    "city_recent": plus("city_recent"),
    "all": plus(*NEW_PATHS, pop_windows=(90, 365, 730, 1825)),
}

PLANS = {
    "curve": SearchPlan(
        variants={"base": base.LINK},
        ref_config=REF,
        grid=[REF],
        stage1_rows=25_000,
        train_sizes=(50_000, 100_000, 200_000, None),
        tune_seeds=(0, 1, 2),
        final_seeds=(0, 1, 2),
        chunk_rows=40_000,
        workers=8,
        prebuild="full",
    ),
    "ablation": SearchPlan(
        variants=VARIANTS,
        ref_config=REF,
        grid=[REF],
        stage1_rows=25_000,
        train_sizes=(),
        tune_seeds=(0, 1, 2),
        final_seeds=(0, 1, 2),
        chunk_rows=3_000,
        workers=4,
        prebuild="stage1",
    ),
}

if __name__ == "__main__":
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--plan", choices=sorted(PLANS), required=True)
    a, rest = p.parse_known_args()
    if "--test" in rest:
        sys.exit("these plans are val only; the final config gets its own plan")
    main(
        "rel-trial",
        "site-sponsor-run",
        PLANS[a.plan],
        base.SRC_FEATURES,
        base.DST_FEATURES,
        argv=rest,
        approach_dir=f"cand-lgbm-v2/{a.plan}",
    )
