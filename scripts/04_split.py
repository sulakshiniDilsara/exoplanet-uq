"""Step 4: split the kept signals into train / val / test, by star.

Run from the project root, after Step 3:
    python scripts/04_split.py

Outputs:
    metadata/split.csv         kepoi_name, kepid, label, split
    metadata/split_info.json   seed, fractions and counts per split

Rule: Step 5 uses only train and val. The test set is opened once, at the end.
"""

import json

import pandas as pd

from exoplanet_uq.config import METADATA_DIR
from exoplanet_uq.data import preprocess as pp
from exoplanet_uq.data import split as sp


def main() -> None:
    processed = pd.read_csv(METADATA_DIR / "processed_signals.csv")
    kept_names = processed.loc[processed["status"] == pp.STATUS_KEPT, "kepoi_name"]

    # Labels come from Step 1's table, the one place where labels live.
    labelled = pd.read_csv(METADATA_DIR / "koi_labelled.csv")
    signals = labelled.loc[
        labelled["kepoi_name"].isin(kept_names), ["kepoi_name", "kepid", "label"]
    ]
    if len(signals) != len(kept_names):
        raise sp.SplitError("Some kept signals are missing from koi_labelled.csv.")

    table = sp.assign_splits(signals).sort_values(["split", "kepid", "kepoi_name"])
    table.to_csv(METADATA_DIR / "split.csv", index=False)

    counts = {}
    for name in sp.SPLITS:
        part = table[table["split"] == name]
        counts[name] = {
            "stars": int(part["kepid"].nunique()),
            "signals": len(part),
            "planets": int(part["label"].sum()),
            "planet_share": round(float(part["label"].mean()), 3),
        }
    info = {
        "seed": sp.SPLIT_SEED,
        "val_fraction": sp.VAL_FRACTION,
        "test_fraction": sp.TEST_FRACTION,
        "split_by": "kepid (star), stratified by has_planet",
        "counts": counts,
    }
    (METADATA_DIR / "split_info.json").write_text(
        json.dumps(info, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
