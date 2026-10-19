"""Step 1: download the KOI catalogue from NASA, check it, and assign labels.

Run from the project root:
    python scripts/01_download_koi.py

Outputs (all in metadata/):
    koi_cumulative_raw.csv   the table exactly as NASA sent it
    koi_labelled.csv         usable rows with label 1 (planet) or 0 (false positive)
    koi_excluded.csv         rows we cannot use, each with a reason
    koi_download_info.json   when and how the table was downloaded
"""

import json
from datetime import datetime, timezone

from exoplanet_uq.config import METADATA_DIR
from exoplanet_uq.data import koi


def main() -> None:
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading the KOI table from the NASA Exoplanet Archive ...")
    text = koi.download_koi_csv()
    downloaded_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    table = koi.parse_koi_csv(text)
    koi.validate_koi_table(table)

    # Save NASA's answer unchanged first. NASA updates this table over
    # time, so this file is our proof of exactly what we used.
    (METADATA_DIR / "koi_cumulative_raw.csv").write_text(text, encoding="utf-8")

    labelled, excluded = koi.assign_labels(table)
    labelled.to_csv(METADATA_DIR / "koi_labelled.csv", index=False)
    excluded.to_csv(METADATA_DIR / "koi_excluded.csv", index=False)

    info = {
        "source": koi.TAP_URL,
        "table": koi.KOI_TABLE,
        "query": koi.build_tap_query(),
        "downloaded_at_utc": downloaded_at,
        "rows_total": len(table),
        "rows_labelled": len(labelled),
        "rows_excluded": len(excluded),
        "planets_label_1": int((labelled["label"] == 1).sum()),
        "false_positives_label_0": int((labelled["label"] == 0).sum()),
        "unique_stars_labelled": int(labelled["kepid"].nunique()),
    }
    (METADATA_DIR / "koi_download_info.json").write_text(
        json.dumps(info, indent=2) + "\n", encoding="utf-8"
    )

    print("\nNASA disposition counts (all rows):")
    print(table["koi_disposition"].value_counts(dropna=False).to_string())
    print("\nWhy rows were excluded:")
    print(excluded["exclusion_reason"].value_counts().to_string())
    print("\nSummary:")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
