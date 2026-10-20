"""Clean the Kepler light curves and turn each signal into fixed length views.

For one star we:
  1. read every quarter file, drop flagged points and NaN,
  2. divide each quarter by its own median and join them (quarter jumps go),
  3. remove upward outliers only (cosmic rays; dips go down, so they stay),
  4. mask the dips of every known signal on the star,
  5. flatten the slow trend, keeping the masked dips untouched.

Then for each signal of that star we fold on its period and make two views:
  global: the whole orbit in GLOBAL_BINS bins,
  local:  a zoom on the dip, LOCAL_WIDTH_DURATIONS dip lengths wide, in LOCAL_BINS bins.

The label is never used here, so planets and false positives are cleaned
in exactly the same way.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# How many numbers each view has. 2001 / 201 are odd, so the dip sits in
# the middle bin.
GLOBAL_BINS = 2001
LOCAL_BINS = 201

# The local view is this many dip lengths wide (2 on each side of the centre).
LOCAL_WIDTH_DURATIONS = 4.0

# Each local bin is this many dip lengths wide, so neighbouring bins overlap.
# Kepler gives one point every 29.4 minutes; for a short dip, 201 touching
# bins are narrower than that, and most would stay empty. Overlapping bins
# (the same value AstroNet used) fix that and also the "strobe" effect,
# when a period is close to a whole number of cadences and every dip is
# sampled at the same places.
LOCAL_BIN_WIDTH_DURATIONS = 0.16

# Mask a little more than the dip itself, because period and time0 have
# small errors that grow over 4 years.
MASK_WIDTH_FACTOR = 1.5

# The flatten window must be much longer than any dip, or it bends into the dip.
FLATTEN_DURATION_FACTOR = 3.0
FLATTEN_MIN_DAYS = 2.0

# Remove points this many standard deviations ABOVE the median.
COSMIC_RAY_SIGMA = 4.0

# The rules agreed in the Step 3 plan.
MIN_TRANSITS = 2
MAX_EMPTY_FRACTION = 0.10

STATUS_KEPT = "kept"
STATUS_EXCLUDED = "excluded"


class PreprocessError(ValueError):
    """Raised when the data cannot be cleaned (for example: no usable points)."""


@dataclass(frozen=True)
class Signal:
    """The numbers we know about one KOI signal."""

    period_days: float
    time0: float  # same clock as the FITS TIME column (BJD - 2454833)
    duration_hours: float

    def __post_init__(self):
        # Bad numbers would give a wrong fold without any error, so stop here.
        values = (self.period_days, self.time0, self.duration_hours)
        if not all(np.isfinite(value) for value in values):
            raise PreprocessError(f"Signal has a missing value: {values}")
        if self.period_days <= 0 or self.duration_hours <= 0:
            raise PreprocessError(f"Period and duration must be positive: {values}")

    @property
    def duration_days(self) -> float:
        return self.duration_hours / 24.0


@dataclass(frozen=True)
class Views:
    """The result for one signal."""

    status: str
    reason: str
    global_view: np.ndarray | None
    local_view: np.ndarray | None
    n_transits: int
    empty_global: int
    empty_local: int


# ---------- one star ----------


def read_star(folder: Path):
    """Read every quarter file of one star into one normalised light curve."""
    import lightkurve as lk  # slow to import, so only when really needed

    paths = sorted(Path(folder).glob("*.fits"))
    if not paths:
        raise PreprocessError(f"No FITS files in {folder}")
    quarters = []
    for path in paths:
        # "default" drops the points the Kepler team flagged as bad.
        quarter = lk.read(path, quality_bitmask="default").remove_nans()
        if len(quarter) > 0:
            quarters.append(quarter)
    if not quarters:
        raise PreprocessError(f"No usable points in {folder}")
    # stitch() divides each quarter by its own median, then joins them.
    return lk.LightCurveCollection(quarters).stitch()


def remove_cosmic_rays(light_curve, sigma: float = COSMIC_RAY_SIGMA):
    """Remove points far ABOVE the median. Points below are never removed."""
    return light_curve.remove_outliers(sigma_lower=np.inf, sigma_upper=sigma)


def transit_mask(time: np.ndarray, signals: list[Signal]) -> np.ndarray:
    """True for every point that is inside (or near) a known dip."""
    mask = np.zeros(len(time), dtype=bool)
    for signal in signals:
        offset = phase_fold(time, signal.period_days, signal.time0)
        mask |= np.abs(offset) < 0.5 * MASK_WIDTH_FACTOR * signal.duration_days
    return mask


def flatten_window(longest_duration_days: float, cadence_days: float) -> int:
    """The flatten window, in points. Always odd (savgol_filter needs that)."""
    days = max(FLATTEN_MIN_DAYS, FLATTEN_DURATION_FACTOR * longest_duration_days)
    points = int(np.ceil(days / cadence_days))
    return points if points % 2 == 1 else points + 1


def flatten_star(light_curve, signals: list[Signal]):
    """Remove the slow trend. Known dips are masked, so the trend ignores them."""
    time = np.asarray(light_curve.time.value, dtype=float)
    cadence_days = float(np.median(np.diff(time)))
    longest = max(signal.duration_days for signal in signals)
    window = flatten_window(longest, cadence_days)
    mask = transit_mask(time, signals)
    return light_curve.flatten(window_length=window, mask=mask)


# ---------- one signal ----------


def phase_fold(time: np.ndarray, period: float, time0: float) -> np.ndarray:
    """Time from the nearest dip centre, in days, between -period/2 and +period/2."""
    return np.mod(time - time0 + 0.5 * period, period) - 0.5 * period


def bin_view(
    x: np.ndarray,
    flux: np.ndarray,
    low: float,
    high: float,
    n_bins: int,
    bin_width: float | None = None,
):
    """Median flux in n_bins bins whose centres are spread evenly from low to high.

    By default a bin is exactly as wide as the gap between centres, so the
    bins touch but do not overlap. A wider bin_width makes neighbouring bins
    share points: then a view stays full even when few points reach it.
    Empty bins are NaN.
    """
    spacing = (high - low) / n_bins
    width = spacing if bin_width is None else max(bin_width, spacing)
    centres = low + spacing * (np.arange(n_bins) + 0.5)

    # Sort once, then find each bin's points with a fast binary search.
    order = np.argsort(x, kind="stable")
    sorted_x = x[order]
    sorted_flux = flux[order]
    starts = np.searchsorted(sorted_x, centres - 0.5 * width, side="left")
    ends = np.searchsorted(sorted_x, centres + 0.5 * width, side="left")

    values = np.full(n_bins, np.nan)
    for number, (start, end) in enumerate(zip(starts, ends, strict=True)):
        if end > start:
            values[number] = np.median(sorted_flux[start:end])
    return values


def fill_empty_bins(values: np.ndarray) -> np.ndarray:
    """Fill NaN bins from their neighbours (straight line between them)."""
    filled = values.copy()
    empty = np.isnan(filled)
    if empty.all():
        raise PreprocessError("Every bin is empty.")
    if empty.any():
        index = np.arange(len(filled))
        filled[empty] = np.interp(index[empty], index[~empty], filled[~empty])
    return filled


def count_transits(time: np.ndarray, signal: Signal) -> int:
    """How many separate dips have at least one data point inside them."""
    offset = phase_fold(time, signal.period_days, signal.time0)
    inside = np.abs(offset) < 0.5 * signal.duration_days
    # Which dip number (0, 1, 2, ...) each in-dip point belongs to.
    epochs = np.round((time[inside] - signal.time0) / signal.period_days)
    return len(np.unique(epochs))


def make_views(time: np.ndarray, flux: np.ndarray, signal: Signal) -> Views:
    """Fold one signal, bin it into a global and a local view, and check the result."""
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    offset = phase_fold(time, signal.period_days, signal.time0)

    global_raw = bin_view(offset / signal.period_days, flux, -0.5, 0.5, GLOBAL_BINS)
    half_width = min(0.5 * LOCAL_WIDTH_DURATIONS * signal.duration_days, 0.5 * signal.period_days)
    local_raw = bin_view(
        offset,
        flux,
        -half_width,
        half_width,
        LOCAL_BINS,
        bin_width=LOCAL_BIN_WIDTH_DURATIONS * signal.duration_days,
    )

    n_transits = count_transits(time, signal)
    empty_global = int(np.isnan(global_raw).sum())
    empty_local = int(np.isnan(local_raw).sum())

    reason = ""
    if n_transits < MIN_TRANSITS:
        reason = f"too_few_transits:{n_transits}"
    elif max(empty_global / GLOBAL_BINS, empty_local / LOCAL_BINS) > MAX_EMPTY_FRACTION:
        reason = f"too_many_empty_bins:global={empty_global},local={empty_local}"
    if reason:
        return Views(STATUS_EXCLUDED, reason, None, None, n_transits, empty_global, empty_local)

    return Views(
        STATUS_KEPT,
        "",
        fill_empty_bins(global_raw).astype(np.float32),
        fill_empty_bins(local_raw).astype(np.float32),
        n_transits,
        empty_global,
        empty_local,
    )
