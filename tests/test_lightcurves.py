"""Tests for picking stars and downloading light curves.

These tests never use the internet. A fake "get" plays the archive: it
answers with a folder page and with tiny FITS files made in memory, so
the tests are fast and always give the same result.
"""

import io

import numpy as np
import pandas as pd
import pytest
import requests
from astropy.io import fits

from exoplanet_uq.data import lightcurves as lc

KEPID = 757450
NAMES = [
    "kplr000757450-2009166043257_llc.fits",
    "kplr000757450-2009259160929_llc.fits",
    "kplr000757450-2009350155506_llc.fits",
]

# ---------- helpers ----------


def fits_bytes(columns=("TIME", lc.FLUX_COLUMN)):
    """A tiny light curve FITS file, as bytes."""
    table = fits.BinTableHDU.from_columns(
        [fits.Column(name=name, format="E", array=np.ones(3)) for name in columns]
    )
    buffer = io.BytesIO()
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(buffer)
    return buffer.getvalue()


GOOD = fits_bytes()
NO_FLUX = fits_bytes(columns=("TIME",))


class FakeResponse:
    def __init__(self, status_code=200, text="", content=b""):
        self.status_code = status_code
        self.text = text
        self.content = content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size):
        for start in range(0, len(self.content), chunk_size):
            yield self.content[start : start + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeArchive:
    """Plays the archive. files maps a file name to the bytes it serves."""

    def __init__(self, files, folder_status=200, fail_times=0, extra_page_text=""):
        self.files = files
        self.folder_status = folder_status
        self.fail_times = fail_times
        self.extra_page_text = extra_page_text
        self.urls = []

    def get(self, url, timeout, stream=False):
        assert timeout > 0  # we must never wait forever
        self.urls.append(url)
        if len(self.urls) <= self.fail_times:
            raise requests.ConnectionError("archive did not answer")
        if url.endswith("/"):
            links = "".join(f'<a href="{name}">{name}</a>\n' for name in self.files)
            return FakeResponse(self.folder_status, text=links + self.extra_page_text)
        name = url.rsplit("/", 1)[1]
        if name not in self.files:
            return FakeResponse(404)
        return FakeResponse(content=self.files[name])


def make_labelled():
    # Star 1: one planet. Star 2: two false positives. Star 3: planet + false positive.
    return pd.DataFrame(
        {
            "kepid": [1, 2, 2, 3, 3],
            "kepoi_name": ["A", "B", "C", "D", "E"],
            "label": [1, 0, 0, 1, 0],
        }
    )


def make_stars(n_with, n_without):
    return pd.DataFrame(
        {
            "kepid": range(1, n_with + n_without + 1),
            "has_planet": [1] * n_with + [0] * n_without,
            "n_signals": 1,
        }
    )


# ---------- validate_kepid, star_dir, star_url ----------


@pytest.mark.parametrize("value", [1, 10797460, np.int64(757450)])
def test_validate_kepid_accepts_positive_whole_numbers(value):
    assert lc.validate_kepid(value) == int(value)


@pytest.mark.parametrize("value", [0, -5, 3.0, "123", "..\\..\\Windows", None, True])
def test_validate_kepid_rejects_anything_else(value):
    with pytest.raises(lc.LightCurveError):
        lc.validate_kepid(value)


def test_star_dir_is_inside_the_raw_folder(tmp_path):
    assert lc.star_dir(tmp_path, KEPID) == tmp_path / "757450"


def test_star_dir_rejects_path_tricks(tmp_path):
    with pytest.raises(lc.LightCurveError):
        lc.star_dir(tmp_path, "../outside")


def test_star_url_uses_the_archive_folder_layout():
    assert lc.star_url(10797460) == f"{lc.ARCHIVE_URL}/0107/010797460/"
    assert lc.star_url(KEPID) == f"{lc.ARCHIVE_URL}/0007/000757450/"
    assert lc.star_url(KEPID).startswith("https://")


# ---------- parse_llc_names ----------


def test_parse_keeps_only_this_stars_long_cadence_files():
    page = (
        '<a href="kplr000757450-2009166043257_llc.fits">ok</a>'
        '<a href="kplr000757450-2009166043257_llc.fits">same file twice</a>'
        '<a href="kplr000757450-2009166043257_slc.fits">short cadence</a>'
        '<a href="kplr000999999-2009166043257_llc.fits">another star</a>'
        '<a href="kplr000757450-20091660_llc.fits">wrong shape</a>'
        '<a href="../../evil_llc.fits">path trick</a>'
    )
    assert lc.parse_llc_names(page, KEPID) == ["kplr000757450-2009166043257_llc.fits"]


def test_parse_an_empty_page_gives_no_files():
    assert lc.parse_llc_names("<html>nothing here</html>", KEPID) == []


# ---------- star_table ----------


def test_star_table_has_one_row_per_star():
    stars = lc.star_table(make_labelled())
    assert stars["kepid"].tolist() == [1, 2, 3]
    assert stars["n_signals"].tolist() == [1, 2, 2]


def test_a_star_with_any_planet_signal_has_a_planet():
    stars = lc.star_table(make_labelled())
    assert stars["has_planet"].tolist() == [1, 0, 1]


def test_star_table_rejects_missing_columns():
    with pytest.raises(lc.LightCurveError, match="label"):
        lc.star_table(pd.DataFrame({"kepid": [1]}))


def test_star_table_rejects_an_empty_table():
    with pytest.raises(lc.LightCurveError, match="empty"):
        lc.star_table(pd.DataFrame(columns=["kepid", "label"]))


# ---------- pick_pilot_stars ----------


def test_pick_keeps_the_planet_ratio():
    picked = lc.pick_pilot_stars(make_stars(40, 60), n_stars=20)
    assert len(picked) == 20
    assert picked["has_planet"].sum() == 8  # 40% of 20


def test_pick_gives_the_same_stars_every_time():
    stars = make_stars(40, 60)
    first = lc.pick_pilot_stars(stars, n_stars=20)
    second = lc.pick_pilot_stars(stars, n_stars=20)
    pd.testing.assert_frame_equal(first, second)


def test_pick_has_no_duplicate_stars():
    picked = lc.pick_pilot_stars(make_stars(40, 60), n_stars=50)
    assert picked["kepid"].is_unique


@pytest.mark.parametrize("n_stars", [0, -1, 101])
def test_pick_rejects_impossible_sizes(n_stars):
    with pytest.raises(lc.LightCurveError, match="n_stars"):
        lc.pick_pilot_stars(make_stars(40, 60), n_stars=n_stars)


# ---------- fits_has_flux ----------


def test_fits_with_flux_column_is_good(tmp_path):
    path = tmp_path / "good.fits"
    path.write_bytes(GOOD)
    assert lc.fits_has_flux(path)


def test_fits_without_flux_column_is_bad(tmp_path):
    path = tmp_path / "noflux.fits"
    path.write_bytes(NO_FLUX)
    assert not lc.fits_has_flux(path)


def test_a_broken_or_missing_file_is_bad(tmp_path):
    path = tmp_path / "broken.fits"
    path.write_bytes(b"this is not a FITS file")
    assert not lc.fits_has_flux(path)
    assert not lc.fits_has_flux(tmp_path / "missing.fits")


# ---------- download_file ----------


def test_download_file_writes_the_file_and_no_part_file(tmp_path):
    archive = FakeArchive({NAMES[0]: GOOD})
    path = tmp_path / NAMES[0]
    lc.download_file(lc.star_url(KEPID) + NAMES[0], path, archive.get)
    assert path.read_bytes() == GOOD
    assert list(tmp_path.iterdir()) == [path]


def test_download_file_stops_a_too_big_file_and_leaves_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(lc, "MAX_FILE_BYTES", 100)
    monkeypatch.setattr(lc, "CHUNK_BYTES", 40)
    archive = FakeArchive({NAMES[0]: b"x" * 500})
    with pytest.raises(lc.LightCurveError, match="bigger"):
        lc.download_file(lc.star_url(KEPID) + NAMES[0], tmp_path / NAMES[0], archive.get)
    assert list(tmp_path.iterdir()) == []  # no half file, no .part file


def test_download_file_raises_on_http_error_and_leaves_nothing(tmp_path):
    archive = FakeArchive({})
    with pytest.raises(requests.HTTPError):
        lc.download_file(lc.star_url(KEPID) + NAMES[0], tmp_path / NAMES[0], archive.get)
    assert list(tmp_path.iterdir()) == []


# ---------- download_star ----------


def test_good_download_is_ok_and_marked_complete(tmp_path):
    archive = FakeArchive({name: GOOD for name in NAMES})
    result = lc.download_star(KEPID, tmp_path, get=archive.get)
    assert result == lc.StarResult(KEPID, lc.STATUS_OK, 3)
    assert lc.is_complete(tmp_path / "757450")
    assert [p.name for p in lc.fits_files(tmp_path / "757450")] == NAMES


def test_a_complete_star_is_skipped(tmp_path):
    lc.download_star(KEPID, tmp_path, get=FakeArchive({name: GOOD for name in NAMES}).get)
    archive = FakeArchive({name: GOOD for name in NAMES})
    result = lc.download_star(KEPID, tmp_path, get=archive.get)
    assert result.status == lc.STATUS_OK
    assert result.note == "already downloaded"
    assert archive.urls == []  # the archive was not asked again


def test_good_files_from_a_cut_run_are_not_downloaded_again(tmp_path):
    folder = tmp_path / "757450"
    folder.mkdir()
    (folder / NAMES[0]).write_bytes(GOOD)  # left from an earlier, cut run
    archive = FakeArchive({name: GOOD for name in NAMES})
    result = lc.download_star(KEPID, tmp_path, get=archive.get)
    assert result.status == lc.STATUS_OK
    assert not any(url.endswith(NAMES[0]) for url in archive.urls)


def test_a_star_with_no_archive_folder_is_no_data(tmp_path):
    archive = FakeArchive({}, folder_status=404)
    result = lc.download_star(KEPID, tmp_path, get=archive.get)
    assert result.status == lc.STATUS_NO_DATA
    assert not (tmp_path / "757450").exists()


def test_a_folder_with_no_long_cadence_files_is_no_data(tmp_path):
    archive = FakeArchive({}, extra_page_text='<a href="x_slc.fits">short only</a>')
    result = lc.download_star(KEPID, tmp_path, get=archive.get)
    assert result.status == lc.STATUS_NO_DATA


def test_bad_files_are_removed_and_star_not_complete(tmp_path):
    archive = FakeArchive({NAMES[0]: GOOD, NAMES[1]: GOOD, NAMES[2]: NO_FLUX})
    result = lc.download_star(KEPID, tmp_path, get=archive.get)
    assert result.status == lc.STATUS_BAD_FILE
    assert result.n_files == 2
    assert len(lc.fits_files(tmp_path / "757450")) == 2  # the bad one was deleted
    assert not lc.is_complete(tmp_path / "757450")


def test_a_short_network_problem_is_retried(tmp_path):
    archive = FakeArchive({name: GOOD for name in NAMES}, fail_times=2)
    result = lc.download_star(KEPID, tmp_path, get=archive.get, wait_seconds=0)
    assert result.status == lc.STATUS_OK


def test_a_long_network_problem_becomes_an_error_not_a_crash(tmp_path):
    archive = FakeArchive({name: GOOD for name in NAMES}, fail_times=99)
    result = lc.download_star(KEPID, tmp_path, get=archive.get, wait_seconds=0)
    assert result.status == lc.STATUS_ERROR
    assert "ConnectionError" in result.note
    assert len(result.note) < 250
    assert len(archive.urls) == 3  # tried exactly 3 times


def test_a_server_error_on_the_folder_is_retried_then_an_error(tmp_path):
    archive = FakeArchive({}, folder_status=503)
    result = lc.download_star(KEPID, tmp_path, get=archive.get, wait_seconds=0)
    assert result.status == lc.STATUS_ERROR
    assert "503" in result.note


def test_a_bad_kepid_stops_before_any_request(tmp_path):
    archive = FakeArchive({})
    with pytest.raises(lc.LightCurveError):
        lc.download_star("../outside", tmp_path, get=archive.get)
    assert archive.urls == []
