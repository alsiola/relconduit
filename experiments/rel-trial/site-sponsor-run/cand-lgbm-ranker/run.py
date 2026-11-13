"""rel-trial/site-sponsor-run: candidates + LightGBM ranker. See README.md."""

from relconduit.features import FeatureConfig
from relconduit.link import LinkConfig, Path, join_edges, table_edges
from relconduit.link_lgbm import main


def ran(db):
    """facility -> sponsor through a study at the facility run by the sponsor."""
    fac = table_edges(db, "facilities_studies", "facility_id", "nct_id")
    spon = table_edges(db, "sponsors_studies", "nct_id", "sponsor_id")
    return join_edges(fac, spon)


def sponsor_trials(db):
    return table_edges(db, "sponsors_studies", "nct_id", "sponsor_id")


def place(col):
    """facility -> its city / state / country (static columns of the facility table)."""

    def edges(db):
        return table_edges(db, "facilities", "facility_id", col)

    return edges


LINK = LinkConfig(
    edges={
        "ran": ran,
        "sponsor_trials": sponsor_trials,
        "city": place("city"),
        "country": place("country"),
        "name": place("name"),
    },
    direct=["ran"],
    n_direct=50,
    paths={
        "covisit": Path(("ran", "~ran", "ran"), n=50, half_life_days=730),
        # Sponsors running trials at other sites in the same city / country: the only signal
        # for the half of the sites with no trial before the seed time.
        "city": Path(("city", "~city", "ran"), n=50, half_life_days=730),
        "country": Path(("country", "~country", "ran"), n=30, window_days=730, max_fanout=5000),
        # Other sites with the same name (e.g. "Novartis Investigative Site").
        "name": Path(("name", "~name", "ran"), n=30, half_life_days=730),
    },
    popularity=["ran", "sponsor_trials"],
    pop_windows=(365, 730, 1825),
    pop_candidates={"ran": (365, 30)},
)
SRC_FEATURES = FeatureConfig(windows=(365, 1460, None), hop2=False)
DST_FEATURES = FeatureConfig(windows=(365, 1460, None), hop2=False)

if __name__ == "__main__":
    main("rel-trial", "site-sponsor-run", LINK, SRC_FEATURES, DST_FEATURES, train_sample=25_000)
