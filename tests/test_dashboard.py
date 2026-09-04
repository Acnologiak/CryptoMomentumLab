"""End-to-end smoke test of the Streamlit app, run headlessly and offline.

`load_prices` is replaced with a synthetic basket, so the whole dashboard —
sidebar, every tab, the simulation with its accelerators switched on — is
exercised without a single network call. Anything that would show up as a
red traceback in the browser fails the test here instead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import INTERVAL  # noqa: F401  (keeps sys.path patched)

from momentumlab import config
from momentumlab.data import PriceMatrix

st_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = st_testing.AppTest

APP = "app.py"
COINS = ["BTC", "ETH", "BNB", "XRP", "SOL", "TRX", "DOGE", "ADA", "LINK", "AVAX"]


def fake_basket() -> PriceMatrix:
    """Three years of 4h candles: a trending BTC and nine noisy alts."""
    rng = np.random.default_rng(11)
    index = pd.date_range("2022-01-01", periods=3 * 365 * 6, freq="4h", tz="UTC")
    n = len(index)
    prices = {}
    for i, coin in enumerate(COINS):
        drift = 0.00003 * (1 + i % 4)
        shocks = rng.normal(drift, 0.012, n)
        prices[coin] = 100.0 * np.exp(np.cumsum(shocks))
    return PriceMatrix(pd.DataFrame(prices, index=index).rename_axis("time"), {})


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setattr("momentumlab.ui.dashboard.load_prices",
                        lambda *args, **kwargs: fake_basket())
    return AppTest.from_file(APP, default_timeout=120)


def multiselect_labelled(run, label):
    """Widgets come back in registration order, so pick this one by its label."""
    return next(w for w in run.multiselect if w.label == label)


def assert_clean(run) -> None:
    assert not run.exception, [str(e) for e in run.exception]
    assert not run.error, [str(e) for e in run.error]


def widget_labelled(elements, label):
    return next(w for w in elements if w.label == label)


def test_cold_start_matches_the_configured_defaults(app):
    """The sidebar and the simulation form must open on `config`'s values.

    Both read the same constants, so this is really a guard against a widget
    drifting back to a hard-coded index.
    """
    run = app.run()
    assert_clean(run)

    assert widget_labelled(run.selectbox, "Інтервал свічок").value == config.DEFAULT_INTERVAL
    assert widget_labelled(run.selectbox, "Біржа").value == config.DEFAULT_EXCHANGE
    assert widget_labelled(run.selectbox, "Котирування").value == config.DEFAULT_QUOTE
    assert widget_labelled(run.radio, "Згладжування ціни").value == config.DEFAULT_SMOOTHING
    assert widget_labelled(run.radio, "Річна нормалізація").value == config.DEFAULT_MOMENTUM_MODE
    assert widget_labelled(run.number_input,
                           "Період згладжування, днів").value == config.DEFAULT_SMOOTHING_DAYS
    assert widget_labelled(run.slider,
                           "Глибина історії, років").value == config.DEFAULT_HISTORY_YEARS
    assert multiselect_labelled(run, "Вікна моментуму (днів)").value == config.DEFAULT_WINDOWS

    assert run.selectbox(key="sim_window").value == config.DEFAULT_SIM_WINDOW
    assert run.multiselect(key="sim_pool").value == config.DEFAULT_SIM_POOL
    assert run.slider(key="sim_k").value == config.DEFAULT_SIM_SLOTS
    assert run.number_input(key="sim_buy").value == config.DEFAULT_SIM_ENTRY_EDGE
    assert run.number_input(key="sim_exit").value == 0.0
    assert run.radio(key="sim_sizing").value == "incremental"
    assert not run.checkbox(key="sim_rotation_on").value
    assert not run.checkbox(key="sim_forecast_on").value
    assert not run.checkbox(key="sim_fast_on").value


def test_app_renders_every_tab_without_errors(app):
    run = app.run()
    assert_clean(run)
    assert len(run.tabs) == 5
    assert run.metric[0].value == str(len(COINS))
    assert run.get("plotly_chart"), "no charts were drawn"


def test_simulation_runs_with_both_accelerators_on(app):
    run = app.run()
    assert_clean(run)

    # switch the predictive exit and the fast-window veto on, then rerun
    run.checkbox(key="sim_forecast_on").set_value(True)
    run.checkbox(key="sim_fast_on").set_value(True)
    run = run.run()
    assert_clean(run)

    expanders = [e.label for e in run.expander]
    assert any("Журнал угод" in label for label in expanders)
    assert any("Ранні виходи" in label for label in expanders)


def test_simulation_survives_target_weight_and_rotation(app):
    run = app.run()
    run.radio(key="sim_sizing").set_value("target_weight")
    run = run.run()
    assert_clean(run)

    run.radio(key="sim_sizing").set_value("incremental")
    run = run.run()
    run.checkbox(key="sim_rotation_on").set_value(True)
    run = run.run()
    assert_clean(run)


def test_thresholds_stay_consistent_when_the_entry_bar_drops(app):
    # The exit bar must never end up above the entry bar, even though the user
    # can lower the entry bar after having raised the exit bar.
    run = app.run()
    run.number_input(key="sim_buy").set_value(200.0)
    run = run.run()
    run.number_input(key="sim_exit").set_value(150.0)
    run = run.run()
    run.number_input(key="sim_buy").set_value(20.0)
    run = run.run()
    assert_clean(run)
    assert run.number_input(key="sim_exit").value <= 20.0


def test_empty_coin_selection_shows_a_hint(app):
    run = app.run()
    multiselect_labelled(run, "Показати").set_value([])
    run = run.run()
    assert_clean(run)
    assert any("Оберіть хоча б одну монету" in info.value for info in run.info)
