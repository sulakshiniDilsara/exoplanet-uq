"""Smoke tests for the Streamlit dashboard (app/dashboard.py).

Streamlit's AppTest runs the page without a browser. Every page is opened
once and must show no error. These tests use the committed model, views and
metadata, so they are skipped if those files are missing.
"""

import pandas as pd
import pytest

from exoplanet_uq.config import DATA_DIR, METADATA_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT
from exoplanet_uq.data.split import SPLITS

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = PROJECT_ROOT / "app" / "dashboard.py"
PAGES = (
    "Overview",
    "Signal Explorer",
    "Uncertainty",
    "Follow-up List",
    "Calibration",
    "Research Evidence",
)
# The first run loads the model and runs MC Dropout, so allow time.
TIMEOUT_SECONDS = 120

pytestmark = pytest.mark.skipif(
    not (
        (DATA_DIR / "models" / "cnn_pilot.pt").is_file()
        and (PROCESSED_DATA_DIR / "views.npz").is_file()
    ),
    reason="needs the committed model and views",
)


def open_page(page: str) -> AppTest:
    app = AppTest.from_file(str(APP), default_timeout=TIMEOUT_SECONDS)
    app.run()
    app.radio(key="nav").set_value(page).run()
    return app


def signal_names(app: AppTest) -> set[str]:
    """The selectbox shows "K00190.01 · NASA: ...", so keep only the name part."""
    return {option.split(" · ")[0] for option in app.selectbox(key="signal").options}


@pytest.mark.parametrize("page", PAGES)
def test_every_page_opens_without_an_error(page):
    app = open_page(page)
    assert not app.exception, app.exception
    assert not app.error, [element.value for element in app.error]


def test_signal_explorer_lists_only_train_and_val_signals():
    split = pd.read_csv(METADATA_DIR / "split.csv")
    names_of = {name: set(split.loc[split["split"] == name, "kepoi_name"]) for name in SPLITS}

    app = open_page("Signal Explorer")
    assert signal_names(app) == names_of["val"]  # validation is the default
    assert not app.warning

    app.radio(key="signal_set").set_value("Training (seen in training)").run()
    assert signal_names(app) == names_of["train"]
    assert app.warning  # the "seen in training" warning is shown
    assert not (signal_names(app) & names_of["test"])


def test_picking_another_signal_updates_the_answer():
    app = open_page("Signal Explorer")
    first = app.metric[0].value
    options = app.selectbox(key="signal").options
    app.selectbox(key="signal").set_value(options[-1].split(" · ")[0]).run()
    assert not app.exception
    assert app.metric[0].value != first  # most vs least uncertain signal
