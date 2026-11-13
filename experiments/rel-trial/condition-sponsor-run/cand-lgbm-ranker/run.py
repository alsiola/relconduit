"""rel-trial/condition-sponsor-run: candidates + LightGBM ranker. See README.md."""

from relconduit.features import FeatureConfig
from relconduit.link import LinkConfig, Path, join_edges, table_edges
from relconduit.link_lgbm import main


def ran(db):
    """condition -> sponsor through a study on the condition run by the sponsor."""
    cond = table_edges(db, "conditions_studies", "condition_id", "nct_id")
    spon = table_edges(db, "sponsors_studies", "nct_id", "sponsor_id")
    return join_edges(cond, spon)


def sponsor_trials(db):
    """study -> sponsor: a sponsor's trials on any condition (its overall activity)."""
    return table_edges(db, "sponsors_studies", "nct_id", "sponsor_id")


LINK = LinkConfig(
    edges={"ran": ran, "sponsor_trials": sponsor_trials},
    direct=["ran"],
    n_direct=100,
    paths={
        # Sponsors of trials on conditions that share sponsors with this one.
        "covisit": Path(("ran", "~ran", "ran"), n=100, half_life_days=730),
        "covisit_recent": Path(("ran", "~ran", "ran"), n=50, window_days=730),
    },
    popularity=["ran", "sponsor_trials"],
    pop_windows=(365, 730, 1825),
    pop_candidates={"sponsor_trials": (365, 50), "ran": (365, 50)},
)
SRC_FEATURES = FeatureConfig(windows=(365, 1460, None), hop2=False)
DST_FEATURES = FeatureConfig(windows=(365, 1460, None), hop2=False)

if __name__ == "__main__":
    main(
        "rel-trial", "condition-sponsor-run", LINK, SRC_FEATURES, DST_FEATURES, train_sample=12_000
    )
