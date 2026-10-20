"""Tests for cleaning light curves and making views.

These tests never read real Kepler files. They build fake light curves
where we already know the answer (where the dip is and how deep it is),
then check the cleaning keeps that answer.
"""

import lightkurve as lk
import numpy as np
import pytest

from exoplanet_uq.data import preprocess as pp

CADENCE = 29.4 / 1440  # Kepler long cadence, in days
DEPTH = 0.001  # 1000 ppm
SIGNAL = pp.Signal(period_days=10.0, time0=5.0, duration_hours=3.0)


def fake_time(days=400.0, gap_every=90.0, gap_days=1.0):
    """Evenly spaced times with a short gap after every 'quarter'."""
    time = np.arange(0.0, days, CADENCE)
    in_gap = np.mod(time, gap_every) > gap_every - gap_days
    return time[~in_gap]


def box_dip(time, signal=SIGNAL, depth=DEPTH):
    """1.0 everywhere, 1 - depth inside each dip."""
    offset = pp.phase_fold(time, signal.period_days, signal.time0)
    return np.where(np.abs(offset) < 0.5 * signal.duration_days, 1.0 - depth, 1.0)


# ---------- Signal ----------


@pytest.mark.parametrize(
    "period, time0, duration",
    [(0.0, 5.0, 3.0), (-1.0, 5.0, 3.0), (10.0, 5.0, 0.0), (np.nan, 5.0, 3.0), (10.0, np.inf, 3.0)],
)
def test_signal_rejects_bad_numbers(period, time0, duration):
    with pytest.raises(pp.PreprocessError):
        pp.Signal(period, time0, duration)


def test_signal_allows_negative_time0():
    assert pp.Signal(10.0, -3.0, 2.0).time0 == -3.0


# ---------- phase_fold ----------


def test_dip_centres_fold_to_zero():
    centres = np.array([5.0, 15.0, 25.0, -5.0, 1005.0])
    np.testing.assert_allclose(pp.phase_fold(centres, 10.0, 5.0), 0.0, atol=1e-9)


def test_fold_gives_time_from_the_nearest_dip():
    offset = pp.phase_fold(np.array([7.5, 2.5, 9.0]), 10.0, 5.0)
    np.testing.assert_allclose(offset, [2.5, -2.5, 4.0])


def test_fold_stays_between_minus_and_plus_half_a_period():
    offset = pp.phase_fold(np.linspace(-100, 100, 10001), 10.0, 5.0)
    assert offset.min() >= -5.0 and offset.max() < 5.0


# ---------- transit_mask ----------


def test_mask_covers_the_dip_and_a_little_more_but_not_far_away():
    time = np.array([5.0, 5.05, 5.09, 5.2, 7.0])  # dip half width 0.0625 d, mask 0.094 d
    np.testing.assert_array_equal(pp.transit_mask(time, [SIGNAL]), [True, True, True, False, False])


def test_mask_covers_every_signal_on_the_star():
    other = pp.Signal(period_days=7.0, time0=1.0, duration_hours=2.0)
    time = np.array([5.0, 1.0, 8.0, 3.0])
    np.testing.assert_array_equal(pp.transit_mask(time, [SIGNAL, other]), [True, True, True, False])


# ---------- flatten_window ----------


def test_flatten_window_is_odd_and_at_least_the_minimum():
    window = pp.flatten_window(longest_duration_days=0.1, cadence_days=CADENCE)
    assert window % 2 == 1
    assert window * CADENCE >= pp.FLATTEN_MIN_DAYS


def test_flatten_window_grows_for_long_dips():
    window = pp.flatten_window(longest_duration_days=61.7 / 24, cadence_days=CADENCE)
    assert window % 2 == 1
    assert window * CADENCE >= pp.FLATTEN_DURATION_FACTOR * 61.7 / 24


# ---------- bin_view ----------


def test_bin_view_takes_the_median_of_each_bin():
    x = np.array([0.1, 0.2, 0.3, 1.5, 1.6])
    flux = np.array([1.0, 5.0, 3.0, 7.0, 9.0])
    np.testing.assert_allclose(pp.bin_view(x, flux, 0.0, 2.0, 2), [3.0, 8.0])


def test_bin_view_leaves_empty_bins_as_nan_and_ignores_points_outside():
    x = np.array([-1.0, 0.1, 2.5, 3.0])
    flux = np.array([9.0, 1.0, 2.0, 9.0])
    values = pp.bin_view(x, flux, 0.0, 3.0, 3)
    assert values[0] == 1.0 and np.isnan(values[1]) and values[2] == 2.0


def test_bin_view_with_no_points_is_all_nan():
    assert np.isnan(pp.bin_view(np.array([]), np.array([]), 0.0, 1.0, 5)).all()


def test_wider_bins_overlap_and_share_points():
    x = np.array([0.4, 0.6])  # one point near the edge of each touching bin
    flux = np.array([1.0, 3.0])
    touching = pp.bin_view(x, flux, 0.0, 1.0, 2)
    wide = pp.bin_view(x, flux, 0.0, 1.0, 2, bin_width=1.0)
    np.testing.assert_allclose(touching, [1.0, 3.0])
    np.testing.assert_allclose(wide, [2.0, 2.0])  # each bin now sees both points


def test_overlapping_bins_fill_a_sparse_view():
    x = np.linspace(-1.0, 1.0, 21)  # 21 points for 201 bins
    flux = np.ones_like(x)
    assert np.isnan(pp.bin_view(x, flux, -1.0, 1.0, 201)).sum() > 150
    assert np.isnan(pp.bin_view(x, flux, -1.0, 1.0, 201, bin_width=0.2)).sum() == 0


# ---------- fill_empty_bins ----------


def test_fill_draws_a_straight_line_inside_and_copies_at_the_edges():
    values = np.array([np.nan, 1.0, np.nan, 3.0, np.nan])
    np.testing.assert_allclose(pp.fill_empty_bins(values), [1.0, 1.0, 2.0, 3.0, 3.0])


def test_fill_does_not_change_the_input_or_full_views():
    values = np.array([1.0, np.nan, 3.0])
    pp.fill_empty_bins(values)
    assert np.isnan(values[1])  # input untouched
    np.testing.assert_array_equal(pp.fill_empty_bins(np.array([1.0, 2.0])), [1.0, 2.0])


def test_fill_refuses_an_all_empty_view():
    with pytest.raises(pp.PreprocessError, match="empty"):
        pp.fill_empty_bins(np.full(4, np.nan))


# ---------- count_transits ----------


def test_count_transits_counts_dips_with_data():
    time = fake_time(days=100.0, gap_every=1e9)  # no gaps: dips at 5, 15, ..., 95
    assert pp.count_transits(time, SIGNAL) == 10


def test_count_transits_skips_dips_that_fall_in_a_gap():
    time = fake_time(days=100.0, gap_every=1e9)
    time = time[(time < 14.0) | (time > 16.0)]  # remove the dip at day 15
    assert pp.count_transits(time, SIGNAL) == 9


# ---------- make_views ----------


def test_long_period_short_dip_keeps_a_full_local_view():
    # Like K05383.01: a 1.65 h dip every 98.5 days, 15 dips in 4 years.
    rare = pp.Signal(period_days=98.5, time0=50.0, duration_hours=1.65)
    time = fake_time(days=1470.0)
    views = pp.make_views(time, box_dip(time, rare), rare)
    assert views.status == pp.STATUS_KEPT
    assert views.empty_local / pp.LOCAL_BINS <= pp.MAX_EMPTY_FRACTION


def test_views_find_the_dip_in_the_middle_at_the_right_depth():
    time = fake_time()
    views = pp.make_views(time, box_dip(time), SIGNAL)
    assert views.status == pp.STATUS_KEPT
    assert views.global_view.shape == (pp.GLOBAL_BINS,)
    assert views.local_view.shape == (pp.LOCAL_BINS,)
    assert views.global_view.dtype == np.float32
    middle = pp.LOCAL_BINS // 2
    assert views.local_view[middle] == pytest.approx(1.0 - DEPTH, abs=1e-6)
    assert views.local_view[0] == pytest.approx(1.0, abs=1e-6)  # far from the dip
    # The box dip covers many bins with the same value, so check its centre.
    dip_bins = np.where(views.global_view < 1.0 - 0.5 * DEPTH)[0]
    assert dip_bins.mean() == pytest.approx(pp.GLOBAL_BINS // 2, abs=1)


def test_a_signal_seen_only_once_is_excluded():
    time = fake_time(days=12.0, gap_every=1e9)  # only the dip at day 5 (day 15 is outside)
    views = pp.make_views(time, box_dip(time), SIGNAL)
    assert views.status == pp.STATUS_EXCLUDED
    assert views.reason == "too_few_transits:1"
    assert views.global_view is None


def test_a_view_with_many_empty_bins_is_excluded():
    time = fake_time(days=400.0, gap_every=1e9)
    offset = pp.phase_fold(time, SIGNAL.period_days, SIGNAL.time0)
    time = time[np.abs(offset) < 1.0]  # keep only 2 days of every 10-day orbit
    views = pp.make_views(time, box_dip(time), SIGNAL)
    assert views.status == pp.STATUS_EXCLUDED
    assert views.reason.startswith("too_many_empty_bins")


def test_a_long_dip_local_view_never_wraps_past_half_a_period():
    wide = pp.Signal(period_days=1.0, time0=0.2, duration_hours=8.0)  # 4 x 8 h > 1 day
    time = fake_time(days=60.0, gap_every=1e9)
    views = pp.make_views(time, box_dip(time, wide), wide)
    assert views.status == pp.STATUS_KEPT
    assert views.local_view[0] == pytest.approx(1.0, abs=1e-6)


# ---------- remove_cosmic_rays and flatten_star (with lightkurve) ----------


def test_cosmic_rays_go_but_dips_stay():
    time = fake_time(days=40.0, gap_every=1e9)
    rng = np.random.default_rng(0)
    flux = box_dip(time) + rng.normal(0, 1e-4, len(time))
    flux[100] += 0.01  # a cosmic ray
    curve = pp.remove_cosmic_rays(lk.LightCurve(time=time, flux=flux))
    assert curve.flux.value.max() < 1.001
    assert curve.flux.value.min() < 1.0 - 0.5 * DEPTH  # the dip is still there


def test_flatten_removes_the_slope_and_keeps_the_dip_depth():
    time = fake_time()
    rng = np.random.default_rng(1)
    slope = 1.0 + 0.01 * np.sin(2 * np.pi * time / 30.0)  # slow wave, 10x deeper than the dip
    flux = box_dip(time) * slope + rng.normal(0, 5e-5, len(time))
    flat = pp.flatten_star(lk.LightCurve(time=time, flux=flux), [SIGNAL])

    views = pp.make_views(flat.time.value, flat.flux.value, SIGNAL)
    assert views.status == pp.STATUS_KEPT
    assert views.local_view[0] == pytest.approx(1.0, abs=2e-4)  # flat outside the dip
    depth = 1.0 - views.local_view[pp.LOCAL_BINS // 2]
    assert depth == pytest.approx(DEPTH, rel=0.1)  # dip kept within 10%


def test_read_star_with_no_files_says_so(tmp_path):
    with pytest.raises(pp.PreprocessError, match="No FITS"):
        pp.read_star(tmp_path)
