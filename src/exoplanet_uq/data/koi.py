"""Download, check and label the Kepler Objects of Interest (KOI) catalogue.

The KOI catalogue is our "marking scheme": it tells us which Kepler
signals are confirmed planets and which are false positives.

Source: NASA Exoplanet Archive, "cumulative" KOI table, read through the
archive's TAP (Table Access Protocol) service.
"""

from __future__ import annotations

import io

import pandas as pd
import requests

TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
KOI_TABLE = "cumulative"

# The columns we ask NASA 
KOI_COLUMNS: tuple[str, ...] = (
    "kepid",  # star ID
    "kepoi_name",  # signal ID (one star can have several)
    "koi_disposition",  # NASA's decision
    "koi_period",  # days between dips
    "koi_time0bk",  # time of the first dip (BJD - 2454833)
    "koi_duration",  # how long one dip lasts, in hours
    "koi_depth",  # how deep the dip is, in ppm
    "koi_model_snr",  # how clear the signal is above the noise
    "koi_fpflag_nt",  # false positive: not transit-like
    "koi_fpflag_ss",  # false positive: eclipsing binary signs
    "koi_fpflag_co",  # false positive: signal from a nearby star
    "koi_fpflag_ec",  # false positive: copy of another signal
)

# Every value NASA may use in koi_disposition. Anything else means the
# table changed and a human must look at it before we continue.
KNOWN_DISPOSITIONS = frozenset({"CONFIRMED", "FALSE POSITIVE", "CANDIDATE", "NOT DISPOSITIONED"})

# Our label rules. CANDIDATE and NOT DISPOSITIONED are left out on
# purpose: nobody knows their true answer yet (see Lesson 2, section 2.2).
LABEL_MAP: dict[str, int] = {"CONFIRMED": 1, "FALSE POSITIVE": 0}

# Without these three values we cannot fold the light curve or mask the
# dips, so a row missing any of them cannot be used.
FOLDING_COLUMNS: tuple[str, ...] = ("koi_period", "koi_time0bk", "koi_duration")


class KoiTableError(ValueError):
    """Raised when the KOI table is not in the shape we expect."""


def build_tap_query(columns: tuple[str, ...] = KOI_COLUMNS) -> str:
    """Build the ADQL query that asks the archive for our columns."""
    return f"select {', '.join(columns)} from {KOI_TABLE}"


def download_koi_csv(timeout_seconds: float = 120.0) -> str:
    """Download the KOI table from NASA as CSV text.

    We check the answer before returning it. If NASA sends an error page
    instead of a table, we stop here instead of passing bad data along.
    """
    response = requests.get(
        TAP_URL,
        params={"query": build_tap_query(), "format": "csv"},
        timeout=timeout_seconds,
    )
    response.raise_for_status()

    text = response.text
    first_line = text.split("\n", 1)[0].strip()
    if not first_line.startswith("kepid"):
        # Show only the start, so a huge error page does not flood the screen.
        raise KoiTableError(f"NASA did not return the KOI table. Answer began with: {text[:200]!r}")
    return text


def parse_koi_csv(text: str) -> pd.DataFrame:
    """Turn CSV text into a table."""
    return pd.read_csv(io.StringIO(text))


def validate_koi_table(table: pd.DataFrame) -> None:
    """Check the table before we trust it. Raise KoiTableError if anything is wrong."""
    if table.empty:
        raise KoiTableError("The KOI table is empty.")

    missing = [column for column in KOI_COLUMNS if column not in table.columns]
    if missing:
        raise KoiTableError(f"The KOI table is missing columns: {missing}")

    if table["kepid"].isna().any():
        raise KoiTableError("Some rows have no star ID (kepid).")

    if table["kepoi_name"].isna().any():
        raise KoiTableError("Some rows have no signal ID (kepoi_name).")

    duplicated = table.loc[table["kepoi_name"].duplicated(), "kepoi_name"]
    if not duplicated.empty:
        raise KoiTableError(f"Duplicate signal IDs: {sorted(duplicated.unique())[:10]}")

    unknown = set(table["koi_disposition"].dropna().unique()) - KNOWN_DISPOSITIONS
    if unknown:
        raise KoiTableError(f"Unknown disposition values: {sorted(unknown)}")


def _exclusion_reason(row: pd.Series) -> str | None:
    """Return why a row cannot be used, or None if it can be used."""
    disposition = row["koi_disposition"]
    if pd.isna(disposition):
        return "no_disposition"
    if disposition not in LABEL_MAP:
        return f"disposition_not_used:{disposition}"
    for column in FOLDING_COLUMNS:
        value = row[column]
        if pd.isna(value):
            return f"missing:{column}"
        # Period and duration must be positive. time0bk can be any number.
        if column != "koi_time0bk" and value <= 0:
            return f"not_positive:{column}"
    return None


def assign_labels(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the table into usable rows (with a label) and excluded rows (with a reason).

    The input table is not changed.
    """
    reasons = table.apply(_exclusion_reason, axis=1)
    usable = reasons.isna()

    labelled = table.loc[usable].copy()
    labelled["label"] = labelled["koi_disposition"].map(LABEL_MAP).astype(int)

    excluded = table.loc[~usable].copy()
    excluded["exclusion_reason"] = reasons[~usable]

    return labelled.reset_index(drop=True), excluded.reset_index(drop=True)
