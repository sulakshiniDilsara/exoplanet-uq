"""Step 2: pick the pilot stars and download their Kepler light curves.

Run from the project root, after Step 1:
    python scripts/02_download_lightcurves.py            # pilot: 150 stars
    python scripts/02_download_lightcurves.py --stars 20 # a smaller test run

You can stop it (Ctrl+C) and run it again: stars that are already
complete are skipped.

Outputs:
    data/raw/kepler/<kepid>/...            the FITS files (not in git)
    metadata/lightcurve_manifest.csv       one row per star: status, files, note
    metadata/lightcurve_download_info.json when and how, with counts and size
"""

import argparse
import json
from datetime import UTC, datetime

import pandas as pd

from exoplanet_uq.config import KEPLER_RAW_DIR, METADATA_DIR
from exoplanet_uq.data import lightcurves as lc

PILOT_STARS = 150


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stars", type=int, default=PILOT_STARS, help="how many stars to pick")
    return parser.parse_args()


def folder_size_mb(folder) -> float:
    total = sum(path.stat().st_size for path in folder.rglob("*") if path.is_file())
    return round(total / 1_000_000, 1)


def main() -> None:
    args = parse_args()
    started_at = datetime.now(UTC).isoformat(timespec="seconds")

    labelled = pd.read_csv(METADATA_DIR / "koi_labelled.csv")
    stars = lc.star_table(labelled)
    picked = lc.pick_pilot_stars(stars, args.stars)
    print(
        f"Picked {len(picked)} of {len(stars)} stars "
        f"({int(picked['has_planet'].sum())} with a planet), seed {lc.PILOT_SEED}."
    )

    KEPLER_RAW_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for number, star in enumerate(picked.itertuples(index=False), start=1):
        result = lc.download_star(star.kepid, KEPLER_RAW_DIR)
        rows.append(
            {
                "kepid": result.kepid,
                "has_planet": star.has_planet,
                "n_signals": star.n_signals,
                "status": result.status,
                "n_files": result.n_files,
                "note": result.note,
            }
        )
        print(f"[{number}/{len(picked)}] {result.kepid}: {result.status} ({result.n_files} files)")

    manifest = pd.DataFrame(rows)
    manifest.to_csv(METADATA_DIR / "lightcurve_manifest.csv", index=False)

    counts = manifest["status"].value_counts()
    info = {
        "source": lc.ARCHIVE_URL,
        "files": "Kepler mission pipeline, long cadence (*_llc.fits)",
        "seed": lc.PILOT_SEED,
        "stars_available": len(stars),
        "stars_picked": len(picked),
        "stars_with_planet": int(picked["has_planet"].sum()),
        "status_counts": {status: int(count) for status, count in counts.items()},
        "files_total": int(manifest["n_files"].sum()),
        "size_mb": folder_size_mb(KEPLER_RAW_DIR),
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (METADATA_DIR / "lightcurve_download_info.json").write_text(
        json.dumps(info, indent=2) + "\n", encoding="utf-8"
    )
    print("\nSummary:")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
