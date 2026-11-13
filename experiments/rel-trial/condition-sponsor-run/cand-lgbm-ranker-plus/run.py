"""rel-trial/condition-sponsor-run: wider candidates, all train rows, tuning, 5 seeds. See README.md."""

import importlib.util
import re
from dataclasses import replace
from pathlib import Path as _P

from relconduit.link import Path, table_edges, token_edges
from relconduit.link_search import SearchPlan, main

_spec = importlib.util.spec_from_file_location(
    "base_run", _P(__file__).parents[1] / "cand-lgbm-ranker" / "run.py"
)
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

STOP = {"of", "and", "the", "in", "with", "to", "or", "a", "for", "on", "by", "type", "disease"}


def cond_study(db):
    """condition -> study on it."""
    return table_edges(db, "conditions_studies", "condition_id", "nct_id")


def words(s: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 2 and w not in STOP]


def cond_word(db):
    """condition -> each word of its MeSH term (static)."""
    return token_edges(db, "conditions", "condition_id", "mesh_term", words)


def sponsor_facility(db):
    """sponsor -> facility of one of its studies, at the later date."""
    from relconduit.link import join_edges, reverse_edges

    ss = table_edges(db, "sponsors_studies", "nct_id", "sponsor_id")
    fs = table_edges(db, "facilities_studies", "nct_id", "facility_id")
    return join_edges(reverse_edges(ss), fs)


PLUS = replace(
    base.LINK,
    edges={
        **base.LINK.edges,
        "cond_study": cond_study,
        "cond_word": cond_word,
        "sponsor_facility": sponsor_facility,
    },
    n_direct=200,
    paths={
        # Wider versions of #12's sources.
        "covisit": Path(("ran", "~ran", "ran"), n=200, half_life_days=730),
        "covisit_recent": Path(("ran", "~ran", "ran"), n=100, window_days=730),
        # New: longer co-occurrence memory.
        "covisit_long": Path(("ran", "~ran", "ran"), n=100, half_life_days=1825),
        # New: sponsors of studies on conditions co-listed with this one in a study.
        "cocond": Path(("cond_study", "~cond_study", "ran"), n=200, half_life_days=730),
        "cocond_recent": Path(
            ("cond_study", "~cond_study", "ran"), n=100, window_days=(None, None, 730)
        ),
        # New: sponsors of conditions whose MeSH term shares a word with this one.
        "words": Path(("cond_word", "~cond_word", "ran"), n=100, half_life_days=730),
        # New: sponsors running studies at the facilities this condition's sponsors use.
        "facility": Path(
            ("ran", "sponsor_facility", "~sponsor_facility"),
            n=100,
            window_days=(None, 730, 730),
            max_fanout=300,
        ),
    },
    pop_candidates={"sponsor_trials": (365, 100), "ran": (365, 100)},
)

REF = {"objective": "binary", "num_leaves": 63, "min_child_samples": 20, "feature_fraction": 0.7}
LR = {**REF, "learning_rate": 0.02}
GRID = [
    REF,  # #12's choice, learning rate 0.05
    LR,
    {**LR, "num_leaves": 127},
    {**LR, "min_child_samples": 100},
    {**LR, "lambda_l2": 10.0},
    {**LR, "feature_fraction": 0.5},
    {**LR, "objective": "lambdarank", "min_child_samples": 100},
    {**LR, "objective": "lambdarank", "min_child_samples": 100, "lambdarank_truncation_level": 40},
]

PLAN = SearchPlan(
    variants={"base": base.LINK, "plus": PLUS},
    ref_config=REF,
    grid=GRID,
    stage1_rows=12_000,
    train_sizes=(24_000, None),
    leader=("Kapso", 0.1228),
)

if __name__ == "__main__":
    main("rel-trial", "condition-sponsor-run", PLAN, base.SRC_FEATURES, base.DST_FEATURES)
