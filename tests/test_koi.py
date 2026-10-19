"""Tests for the KOI download, check and label code.

These tests never use the internet. They build small fake tables, so they
are fast and always give the same result.
"""

import pandas as pd
import pytest
import requests

from exoplanet_uq.data import koi


def make_row(**changes):
    """One valid KOI row. Pass keyword arguments to change any field."""
    row = {
        "kepid": 1001,
        "kepoi_name": "K00001.01",
        "koi_disposition": "CONFIRMED",
        "koi_period": 3.5,
        "koi_time0bk": 135.2,
        "koi_duration": 2.8,
        "koi_depth": 800.0,
        "koi_model_snr": 25.0,
        "koi_fpflag_nt": 0,
        "koi_fpflag_ss": 0,
        "koi_fpflag_co": 0,
        "koi_fpflag_ec": 0,
    }
    row.update(changes)
    return row


def make_table(*rows):
    return pd.DataFrame(list(rows))


# ---------- build_tap_query ----------


def test_query_asks_for_every_column_from_the_cumulative_table():
    query = koi.build_tap_query()
    assert query.startswith("select kepid, kepoi_name")
    assert query.endswith("from cumulative")
    for column in koi.KOI_COLUMNS:
        assert column in query


# ---------- parse_koi_csv ----------


def test_parse_reads_csv_text_into_a_table():
    text = "kepid,kepoi_name\n1001,K00001.01\n1002,K00002.01\n"
    table = koi.parse_koi_csv(text)
    assert list(table.columns) == ["kepid", "kepoi_name"]
    assert len(table) == 2


# ---------- validate_koi_table ----------


def test_validate_accepts_a_good_table():
    koi.validate_koi_table(make_table(make_row()))  # no error means pass


def test_validate_rejects_an_empty_table():
    empty = pd.DataFrame(columns=list(koi.KOI_COLUMNS))
    with pytest.raises(koi.KoiTableError, match="empty"):
        koi.validate_koi_table(empty)


def test_validate_rejects_a_missing_column():
    table = make_table(make_row()).drop(columns=["koi_period"])
    with pytest.raises(koi.KoiTableError, match="koi_period"):
        koi.validate_koi_table(table)


def test_validate_rejects_an_unknown_disposition():
    table = make_table(make_row(koi_disposition="MAYBE PLANET"))
    with pytest.raises(koi.KoiTableError, match="MAYBE PLANET"):
        koi.validate_koi_table(table)


def test_validate_rejects_duplicate_signal_ids():
    table = make_table(make_row(), make_row(kepid=2002))  # same kepoi_name twice
    with pytest.raises(koi.KoiTableError, match="Duplicate"):
        koi.validate_koi_table(table)


def test_validate_rejects_a_missing_star_id():
    table = make_table(make_row(kepid=None))
    with pytest.raises(koi.KoiTableError, match="kepid"):
        koi.validate_koi_table(table)


# ---------- assign_labels ----------


def test_confirmed_is_1_and_false_positive_is_0():
    table = make_table(
        make_row(kepoi_name="K00001.01", koi_disposition="CONFIRMED"),
        make_row(kepoi_name="K00002.01", koi_disposition="FALSE POSITIVE"),
    )
    labelled, excluded = koi.assign_labels(table)
    assert labelled["label"].tolist() == [1, 0]
    assert labelled["label"].dtype.kind == "i"  # whole numbers, not 1.0 / 0.0
    assert excluded.empty


@pytest.mark.parametrize("disposition", ["CANDIDATE", "NOT DISPOSITIONED"])
def test_unknown_answers_are_excluded_with_a_reason(disposition):
    labelled, excluded = koi.assign_labels(make_table(make_row(koi_disposition=disposition)))
    assert labelled.empty
    assert excluded["exclusion_reason"].tolist() == [f"disposition_not_used:{disposition}"]


@pytest.mark.parametrize("column", ["koi_period", "koi_time0bk", "koi_duration"])
def test_rows_missing_folding_values_are_excluded(column):
    labelled, excluded = koi.assign_labels(make_table(make_row(**{column: None})))
    assert labelled.empty
    assert excluded["exclusion_reason"].tolist() == [f"missing:{column}"]


@pytest.mark.parametrize("column", ["koi_period", "koi_duration"])
def test_zero_or_negative_period_or_duration_is_excluded(column):
    labelled, excluded = koi.assign_labels(make_table(make_row(**{column: 0.0})))
    assert labelled.empty
    assert excluded["exclusion_reason"].tolist() == [f"not_positive:{column}"]


def test_negative_time0bk_is_allowed():
    # time0bk is a time on a shifted clock, so a negative value is valid.
    labelled, _ = koi.assign_labels(make_table(make_row(koi_time0bk=-5.0)))
    assert len(labelled) == 1


def test_assign_labels_does_not_change_the_input_table():
    table = make_table(make_row(), make_row(kepoi_name="K00002.01", koi_disposition="CANDIDATE"))
    before = table.copy()
    koi.assign_labels(table)
    pd.testing.assert_frame_equal(table, before)


def test_every_row_ends_up_in_exactly_one_output():
    table = make_table(
        make_row(kepoi_name="A", koi_disposition="CONFIRMED"),
        make_row(kepoi_name="B", koi_disposition="CANDIDATE"),
        make_row(kepoi_name="C", koi_disposition="FALSE POSITIVE", koi_period=None),
        make_row(kepoi_name="D", koi_disposition="FALSE POSITIVE"),
    )
    labelled, excluded = koi.assign_labels(table)
    assert sorted(labelled["kepoi_name"]) == ["A", "D"]
    assert sorted(excluded["kepoi_name"]) == ["B", "C"]


# ---------- download_koi_csv (internet replaced by a fake) ----------


class FakeResponse:
    def __init__(self, text, status_code=200):
        self.text = text
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


def test_download_returns_the_table_text(monkeypatch):
    good = "kepid,kepoi_name\n1001,K00001.01\n"
    monkeypatch.setattr(koi.requests, "get", lambda *args, **kwargs: FakeResponse(good))
    assert koi.download_koi_csv() == good


def test_download_rejects_an_error_page(monkeypatch):
    page = "<html>Service temporarily unavailable</html>"
    monkeypatch.setattr(koi.requests, "get", lambda *args, **kwargs: FakeResponse(page))
    with pytest.raises(koi.KoiTableError, match="did not return the KOI table"):
        koi.download_koi_csv()


def test_download_raises_on_http_error(monkeypatch):
    monkeypatch.setattr(koi.requests, "get", lambda *args, **kwargs: FakeResponse("", 503))
    with pytest.raises(requests.HTTPError):
        koi.download_koi_csv()


def test_download_always_sets_a_timeout(monkeypatch):
    seen = {}

    def fake_get(url, params, timeout):
        seen["url"], seen["timeout"] = url, timeout
        return FakeResponse("kepid\n1\n")

    monkeypatch.setattr(koi.requests, "get", fake_get)
    koi.download_koi_csv()
    assert seen["url"].startswith("https://")
    assert seen["timeout"] > 0
