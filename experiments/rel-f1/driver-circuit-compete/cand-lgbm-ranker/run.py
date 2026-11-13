"""rel-f1/driver-circuit-compete: candidates + LightGBM ranker. See README.md."""

from relconduit.features import FeatureConfig
from relconduit.link import LinkConfig, Path, join_edges, table_edges
from relconduit.link_lgbm import main


def raced(db):
    """driver -> circuit, one edge per race result, at the race's date."""
    results = table_edges(db, "results", "driverId", "raceId")
    races = table_edges(db, "races", "raceId", "circuitId")
    return join_edges(results, races)


def calendar(db):
    """race -> circuit: each race on the calendar (popularity of a circuit, any driver)."""
    return table_edges(db, "races", "raceId", "circuitId")


LINK = LinkConfig(
    edges={"raced": raced, "calendar": calendar},
    direct=["raced"],
    n_direct=77,
    paths={
        # Circuits raced by drivers who raced where this driver did (recent seasons count more).
        "covisit": Path(("raced", "~raced", "raced"), n=77, half_life_days=730),
    },
    popularity=["raced", "calendar"],
    pop_windows=(365, 730, 1825),
    # Every circuit on the calendar in the last three seasons (there are 77 circuits in all).
    pop_candidates={"calendar": (1095, 77)},
)
FEATURES = FeatureConfig(windows=(365, 1095, None))

if __name__ == "__main__":
    main("rel-f1", "driver-circuit-compete", LINK, FEATURES, FEATURES)
