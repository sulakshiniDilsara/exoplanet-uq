"""Pick stars and download their Kepler light curves from the MAST archive.

The KOI table (Step 1) gives us the labels. This module gets the light
curves: for each chosen star, every long cadence quarter file that the
Kepler mission pipeline made.

We read the files straight from the MAST archive web folders, one folder
per star. (The MAST search service that lightkurve uses was too slow to
be usable when this was written: one star took more than 6 minutes.)

Files go to data/raw/kepler/<kepid>/. When all files of a star are
downloaded and checked, we write a COMPLETE marker file in its folder, so
a later run can skip that star (resume after a cut or a crash).
"""

from __future__ import annotations

import numbers
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests
from astropy.io import fits

# The official Kepler mission light curve files, one web folder per star.
ARCHIVE_URL = "https://archive.stsci.edu/missions/kepler/lightcurves"

# Seconds to wait for the archive before we give up on one request.
TIMEOUT_SECONDS = 60.0

# One long cadence quarter file is a few MB. Anything far bigger is not
# what we asked for, so we stop instead of filling the disk.
MAX_FILE_BYTES = 50_000_000
CHUNK_BYTES = 1_000_000

# The brightness column we will use in Step 3. NASA has already removed
# some telescope errors from it.
FLUX_COLUMN = "PDCSAP_FLUX"

# Written last, only when every file of a star is downloaded and checked.
COMPLETE_MARKER = "COMPLETE"

# Fixed seed, so everyone who runs the pilot gets the same stars.
PILOT_SEED = 42

# The status values one star can end with in the manifest.
STATUS_OK = "ok"
STATUS_NO_DATA = "no_data"
STATUS_BAD_FILE = "bad_file"
STATUS_ERROR = "error"


class LightCurveError(ValueError):
    """Raised when an input is not safe or not in the shape we expect."""


@dataclass(frozen=True)
class StarResult:
    """What happened when we tried to download one star."""

    kepid: int
    status: str
    n_files: int
    note: str = ""


def validate_kepid(value: object) -> int:
    """Return value as a positive int, or raise LightCurveError.

    The kepid is used in a web address and in a folder name, so we only
    allow plain positive whole numbers. Text like "..\\x" can never pass.
    """
    # bool is a kind of int in Python (True == 1), so reject it on purpose.
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise LightCurveError(f"kepid must be a whole number, got {value!r}")
    if value <= 0:
        raise LightCurveError(f"kepid must be positive, got {value!r}")
    return int(value)


def star_table(labelled: pd.DataFrame) -> pd.DataFrame:
    """Turn the signal table into one row per star.

    A star "has a planet" if at least one of its signals has label 1.
    """
    missing = {"kepid", "label"} - set(labelled.columns)
    if missing:
        raise LightCurveError(f"The labelled table is missing columns: {sorted(missing)}")
    if labelled.empty:
        raise LightCurveError("The labelled table is empty.")

    stars = (
        labelled.groupby("kepid")
        .agg(has_planet=("label", "max"), n_signals=("label", "size"))
        .reset_index()
    )
    stars["has_planet"] = stars["has_planet"].astype(int)
    return stars


def pick_pilot_stars(stars: pd.DataFrame, n_stars: int, seed: int = PILOT_SEED) -> pd.DataFrame:
    """Pick n_stars at random, keeping the planet / no-planet ratio (stratified).

    The same stars, n_stars and seed always give the same result.
    """
    if not 0 < n_stars <= len(stars):
        raise LightCurveError(f"n_stars must be between 1 and {len(stars)}, got {n_stars}")

    share_with_planet = stars["has_planet"].mean()
    n_with_planet = round(n_stars * share_with_planet)
    n_without_planet = n_stars - n_with_planet

    with_planet = stars[stars["has_planet"] == 1]
    without_planet = stars[stars["has_planet"] == 0]
    picked = pd.concat(
        [
            with_planet.sample(n=n_with_planet, random_state=seed),
            without_planet.sample(n=n_without_planet, random_state=seed),
        ]
    )
    return picked.sort_values("kepid").reset_index(drop=True)


def star_dir(raw_dir: Path, kepid: object) -> Path:
    """The folder for one star's files: raw_dir/<kepid>."""
    return Path(raw_dir) / str(validate_kepid(kepid))


def star_url(kepid: object) -> str:
    """The archive web folder of one star, e.g. .../0107/010797460/."""
    padded = f"{validate_kepid(kepid):09d}"
    return f"{ARCHIVE_URL}/{padded[:4]}/{padded}/"


def parse_llc_names(page: str, kepid: int) -> list[str]:
    """Pick this star's long cadence file names out of the folder page.

    The page comes from the internet, so we trust nothing in it: only names
    with exactly the shape kplr<kepid>-<13 digits>_llc.fits are kept.
    """
    pattern = re.compile(rf"kplr{kepid:09d}-\d{{13}}_llc\.fits")
    return sorted(set(pattern.findall(page)))


def fits_files(folder: Path) -> list[Path]:
    """Every FITS file under folder, in a fixed order."""
    return sorted(Path(folder).rglob("*.fits"))


def fits_has_flux(path: Path) -> bool:
    """True if the file opens as FITS and its light curve table has FLUX_COLUMN.

    A file that was cut off half way, or is not FITS at all, gives False.
    """
    try:
        with fits.open(path) as hdul:
            columns = hdul[1].columns.names
    except Exception:  # any read problem means the file is not usable
        return False
    return FLUX_COLUMN in columns


def is_complete(folder: Path) -> bool:
    """True if this star was fully downloaded and checked before."""
    return (Path(folder) / COMPLETE_MARKER).is_file()


def list_llc_files(kepid: int, get: Callable = requests.get) -> list[str]:
    """Ask the archive which long cadence files exist for this star.

    An empty list means the archive has no folder (or no files) for it.
    """
    response = get(star_url(kepid), timeout=TIMEOUT_SECONDS)
    if response.status_code == 404:
        return []
    response.raise_for_status()
    return parse_llc_names(response.text, kepid)


def download_file(url: str, path: Path, get: Callable = requests.get) -> None:
    """Download one file safely.

    We write to "<name>.part" first and rename only at the end, so a cut
    download never looks like a finished file.
    """
    part = path.with_name(path.name + ".part")
    try:
        with get(url, stream=True, timeout=TIMEOUT_SECONDS) as response:
            response.raise_for_status()
            size = 0
            with open(part, "wb") as out:
                for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
                    size += len(chunk)
                    if size > MAX_FILE_BYTES:
                        raise LightCurveError(f"{url} is bigger than {MAX_FILE_BYTES} bytes")
                    out.write(chunk)
        part.replace(path)
    finally:
        # After a success the .part file is already renamed, so this only
        # cleans up after a failure.
        part.unlink(missing_ok=True)


def download_star(
    kepid: object,
    raw_dir: Path,
    get: Callable = requests.get,
    attempts: int = 3,
    wait_seconds: float = 10.0,
) -> StarResult:
    """Download and check every long cadence file of one star.

    One star failing never stops the whole run: the problem is returned
    as a status with a note, and the script writes it to the manifest.
    """
    kepid = validate_kepid(kepid)
    folder = star_dir(raw_dir, kepid)

    if is_complete(folder):
        return StarResult(kepid, STATUS_OK, len(fits_files(folder)), "already downloaded")

    last_error = ""
    for attempt in range(1, attempts + 1):
        try:
            names = list_llc_files(kepid, get)
            if not names:
                return StarResult(kepid, STATUS_NO_DATA, 0, "archive has no long cadence file")
            folder.mkdir(parents=True, exist_ok=True)
            for name in names:
                path = folder / name
                # Files from an earlier, cut run are kept if they are good.
                if not fits_has_flux(path):
                    download_file(star_url(kepid) + name, path, get)
            return _check_star_files(kepid, folder)
        except Exception as error:  # network and archive errors come in many types
            # Keep the note short, so one long error page does not fill the manifest.
            last_error = f"{type(error).__name__}: {error}"[:200]
            if attempt < attempts:
                time.sleep(wait_seconds)

    return StarResult(kepid, STATUS_ERROR, 0, f"failed after {attempts} tries: {last_error}")


def _check_star_files(kepid: int, folder: Path) -> StarResult:
    """Check the downloaded files. Write the COMPLETE marker only if all are good."""
    files = fits_files(folder)
    bad = [path for path in files if not fits_has_flux(path)]
    for path in bad:
        # A broken file must not reach Step 3. Deleting it also means the
        # next run downloads it again.
        path.unlink()
    if bad:
        good = len(files) - len(bad)
        return StarResult(kepid, STATUS_BAD_FILE, good, f"{len(bad)} bad file(s) removed")

    (folder / COMPLETE_MARKER).write_text(f"{len(files)} files\n", encoding="utf-8")
    return StarResult(kepid, STATUS_OK, len(files))
