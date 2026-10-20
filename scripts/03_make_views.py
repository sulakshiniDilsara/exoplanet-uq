"""Step 3: clean the light curves and make a global and a local view per signal.

Run from the project root, after Step 2:
    python scripts/03_make_views.py
    python scripts/03_make_views.py --no-plots    # faster, no pictures

No internet is needed: it reads the files Step 2 downloaded.

Outputs:
    data/processed/views.npz             kepoi_name, global (N x 2001), local (N x 201)
    data/processed/plots/<kepoi>.png     raw / flat / global / local, to check by eye
    metadata/processed_signals.csv       one row per signal: kept or excluded, and why
    metadata/processed_info.json         settings and counts
"""

import argparse
import json
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")  # draw to files only, no window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from exoplanet_uq.config import KEPLER_RAW_DIR, METADATA_DIR, PROCESSED_DATA_DIR  # noqa: E402
from exoplanet_uq.data import lightcurves as lc  # noqa: E402
from exoplanet_uq.data import preprocess as pp  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-plots", action="store_true", help="skip the pictures")
    return parser.parse_args()


def to_signal(row) -> pp.Signal:
    return pp.Signal(row.koi_period, row.koi_time0bk, row.koi_duration)


def mask_signals(raw_rows: pd.DataFrame) -> list[pp.Signal]:
    """Every known signal on a star (CANDIDATE too) that has usable numbers."""
    signals = []
    for row in raw_rows.itertuples(index=False):
        try:
            signals.append(to_signal(row))
        except pp.PreprocessError:
            continue  # a KOI with a missing period cannot be masked; skip it
    return signals


def plot_signal(path, name, label, raw, flat, mask, views, signal) -> None:
    """Four small pictures of one signal, to check the cleaning by eye."""
    fig, axes = plt.subplots(4, 1, figsize=(10, 10))
    raw_time = raw.time.value
    axes[0].plot(raw_time, raw.flux.value, ",k")
    axes[0].set_title(f"{name}  (label {label})  raw, each quarter / its median")
    flat_time = flat.time.value
    axes[1].plot(flat_time, flat.flux.value, ",k")
    axes[1].plot(flat_time[mask], flat.flux.value[mask], ",r")
    axes[1].set_title("flattened (red = masked dips)")
    axes[2].plot(np.linspace(-0.5, 0.5, pp.GLOBAL_BINS), views.global_view, ".k", ms=2)
    axes[2].set_title(f"global view ({pp.GLOBAL_BINS} bins, x = phase)")
    half = min(0.5 * pp.LOCAL_WIDTH_DURATIONS * signal.duration_days, 0.5 * signal.period_days)
    axes[3].plot(np.linspace(-half, half, pp.LOCAL_BINS) * 24, views.local_view, ".k", ms=3)
    axes[3].set_title(f"local view ({pp.LOCAL_BINS} bins, x = hours from dip centre)")
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    manifest = pd.read_csv(METADATA_DIR / "lightcurve_manifest.csv")
    labelled = pd.read_csv(METADATA_DIR / "koi_labelled.csv")
    raw_koi = pd.read_csv(METADATA_DIR / "koi_cumulative_raw.csv")
    stars = manifest.loc[manifest["status"] == lc.STATUS_OK, "kepid"].tolist()

    plot_dir = PROCESSED_DATA_DIR / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    rows, names, global_views, local_views = [], [], [], []
    for number, kepid in enumerate(stars, start=1):
        star_signals = labelled[labelled["kepid"] == kepid]
        try:
            masks = mask_signals(raw_koi[raw_koi["kepid"] == kepid])
            raw = pp.remove_cosmic_rays(pp.read_star(lc.star_dir(KEPLER_RAW_DIR, kepid)))
            flat = pp.flatten_star(raw, masks)
            time = np.asarray(flat.time.value, dtype=float)
            flux = np.asarray(flat.flux.value, dtype=float)
            mask = pp.transit_mask(time, masks)
        except Exception as error:  # one broken star must not stop the others
            note = f"star_error:{type(error).__name__}: {error}"[:200]
            for row in star_signals.itertuples(index=False):
                rows.append(
                    {
                        "kepoi_name": row.kepoi_name,
                        "kepid": kepid,
                        "label": row.label,
                        "status": pp.STATUS_EXCLUDED,
                        "reason": note,
                    }
                )
            print(f"[{number}/{len(stars)}] {kepid}: ERROR {note}")
            continue

        for row in star_signals.itertuples(index=False):
            signal = to_signal(row)
            views = pp.make_views(time, flux, signal)
            rows.append(
                {
                    "kepoi_name": row.kepoi_name,
                    "kepid": kepid,
                    "label": row.label,
                    "status": views.status,
                    "reason": views.reason,
                    "n_points": len(time),
                    "n_transits": views.n_transits,
                    "empty_global": views.empty_global,
                    "empty_local": views.empty_local,
                }
            )
            if views.status == pp.STATUS_KEPT:
                names.append(row.kepoi_name)
                global_views.append(views.global_view)
                local_views.append(views.local_view)
                if not args.no_plots:
                    path = plot_dir / f"{row.kepoi_name}.png"
                    plot_signal(path, row.kepoi_name, row.label, raw, flat, mask, views, signal)
        print(f"[{number}/{len(stars)}] {kepid}: {len(star_signals)} signal(s)")

    table = pd.DataFrame(rows)
    table.to_csv(METADATA_DIR / "processed_signals.csv", index=False)
    np.savez(
        PROCESSED_DATA_DIR / "views.npz",
        kepoi_name=np.array(names, dtype=str),
        global_view=np.array(global_views, dtype=np.float32),
        local_view=np.array(local_views, dtype=np.float32),
    )

    kept = table[table["status"] == pp.STATUS_KEPT]
    info = {
        "global_bins": pp.GLOBAL_BINS,
        "local_bins": pp.LOCAL_BINS,
        "local_width_durations": pp.LOCAL_WIDTH_DURATIONS,
        "local_bin_width_durations": pp.LOCAL_BIN_WIDTH_DURATIONS,
        "mask_width_factor": pp.MASK_WIDTH_FACTOR,
        "flatten_duration_factor": pp.FLATTEN_DURATION_FACTOR,
        "flatten_min_days": pp.FLATTEN_MIN_DAYS,
        "cosmic_ray_sigma": pp.COSMIC_RAY_SIGMA,
        "min_transits": pp.MIN_TRANSITS,
        "max_empty_fraction": pp.MAX_EMPTY_FRACTION,
        "stars": len(stars),
        "signals": len(table),
        "kept": len(kept),
        "kept_planets": int(kept["label"].sum()),
        "excluded": int((table["status"] == pp.STATUS_EXCLUDED).sum()),
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (METADATA_DIR / "processed_info.json").write_text(
        json.dumps(info, indent=2) + "\n", encoding="utf-8"
    )
    print("\nWhy signals were excluded:")
    print(table["reason"].replace("", np.nan).dropna().value_counts().to_string())
    print("\nSummary:")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
