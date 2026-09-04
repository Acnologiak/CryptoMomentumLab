"""Synthetic-data checks for the annualization math.

Run:  python -m pytest tests -q     (or simply: python tests/test_analytics.py)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from conftest import INTERVAL, bar_index, growth_frame

from momentumlab.analytics import (
    MomentumCalculator,
    MomentumMode,
    RankingAnalyzer,
    Smoothing,
    decimate,
    smooth_prices,
)
from momentumlab.timeframe import Timeframe, window_label


def calculator(mode=MomentumMode.COMPOUND, smoothing=Smoothing.NONE) -> MomentumCalculator:
    return MomentumCalculator(INTERVAL, smoothing=smoothing, mode=mode)


def test_all_windows_agree_on_constant_growth():
    """A steady trend must annualize to the same number on every window."""
    prices = growth_frame(900, X=0.001)  # +0.1% per day
    expected = ((1.001 ** 365) - 1) * 100
    for window in (7, 30, 90, 365):
        got = calculator().curve(prices, window)["X"].iloc[-1]
        assert abs(got - expected) < 1e-6, (window, got, expected)


def test_log_mode_matches_definition():
    prices = growth_frame(900, X=0.001)
    expected = np.log(1.001) * 365 * 100
    got = calculator(MomentumMode.LOG).curve(prices, 30)["X"].iloc[-1]
    assert abs(got - expected) < 1e-6, (got, expected)


def test_flat_price_gives_zero():
    idx = bar_index(84)
    flat = pd.DataFrame({"X": np.full(len(idx), 42.0)}, index=idx)
    out = calculator().curve(flat, 30)["X"].dropna()
    assert np.allclose(out.values, 0.0)


def test_window_shorter_than_history_leaves_leading_nan():
    prices = growth_frame(40, X=0.001)
    out = calculator(MomentumMode.LOG).curve(prices, 30)["X"]
    shift = Timeframe(INTERVAL).bars(30)
    assert out.iloc[:shift].isna().all()
    assert out.iloc[shift:].notna().all()


def test_window_longer_than_history_is_all_nan():
    prices = growth_frame(40, X=0.001)
    assert calculator(MomentumMode.LOG).curve(prices, 365)["X"].isna().all()


def test_smoothing_lengths_are_interval_independent():
    assert Timeframe("4h").bars(7) == 42
    assert Timeframe("1h").bars(7) == 168
    assert Timeframe("1d").bars(7) == 7

    prices = growth_frame(900, X=0.001)
    ema = smooth_prices(prices, Smoothing.EMA, 7, INTERVAL)["X"]
    sma = smooth_prices(prices, Smoothing.SMA, 7, INTERVAL)["X"]
    expected_warmup = Timeframe(INTERVAL).bars(7) - 1
    assert ema.isna().sum() == sma.isna().sum() == expected_warmup
    assert smooth_prices(prices, Smoothing.NONE, 7, INTERVAL).equals(prices)


def test_zlema_has_a_clean_warmup_split():
    prices = growth_frame(900, X=0.001)
    span = Timeframe(INTERVAL).bars(7)
    zlema = smooth_prices(prices, Smoothing.ZLEMA, 7, INTERVAL)["X"]

    first_valid = zlema.notna().idxmax()
    # a contiguous NaN prefix, then everything valid (no holes past warmup)
    assert zlema.loc[:first_valid].iloc[:-1].isna().all()
    assert zlema.loc[first_valid:].notna().all()
    assert zlema.index.get_loc(first_valid) >= span  # at least a full EMA span


def test_zlema_tracks_a_linear_trend_with_less_lag_than_ema():
    # On a straight ramp a plain EMA sits a fixed distance *behind* the price;
    # ZLEMA's de-lagging should sit far closer to the true line.
    idx = bar_index(400)
    ramp = pd.DataFrame({"X": 100.0 + 0.05 * np.arange(len(idx))}, index=idx)
    ema = smooth_prices(ramp, Smoothing.EMA, 10, INTERVAL)["X"]
    zlema = smooth_prices(ramp, Smoothing.ZLEMA, 10, INTERVAL)["X"]

    tail = slice(-500, None)
    ema_lag = (ramp["X"] - ema).iloc[tail].abs().mean()
    zlema_lag = (ramp["X"] - zlema).iloc[tail].abs().mean()
    assert zlema_lag < 0.1 * ema_lag, (zlema_lag, ema_lag)


def test_panel_and_snapshot_shapes():
    prices = growth_frame(900, A=0.001, B=0.0005)
    panel = MomentumCalculator(INTERVAL, smoothing=Smoothing.EMA).panel(prices, [7, 30, 90])

    assert panel.windows == [7.0, 30.0, 90.0]
    assert 30.0 in panel and len(panel) == 3
    snapshot = panel.snapshot()
    assert list(snapshot.index) == ["A", "B"]
    assert list(snapshot.columns) == ["7д", "1м", "3м"]
    assert (snapshot.loc["A"] > snapshot.loc["B"]).all()


def test_panel_before_cuts_the_tail_off():
    prices = growth_frame(900, A=0.001)
    panel = MomentumCalculator(INTERVAL).panel(prices, [30])
    cutoff = prices.index[500]
    assert panel.before(cutoff)[30.0].index[-1] < cutoff


def test_ranking_and_leader():
    idx = bar_index(50)
    n = len(idx)
    momentum = pd.DataFrame({"A": np.linspace(0, 100, n),
                             "B": np.linspace(100, 0, n)}, index=idx)
    analyzer = RankingAnalyzer(momentum)

    ranks = analyzer.ranks()
    assert ranks["A"].iloc[-1] == 1 and ranks["B"].iloc[-1] == 2
    assert analyzer.leader().iloc[-1] == "A"
    assert abs(analyzer.leader_shares().sum() - 100) < 1e-9


def test_decimate_keeps_endpoints():
    idx = bar_index(834)
    frame = pd.DataFrame({"X": np.arange(float(len(idx)))}, index=idx)
    small = decimate(frame, 500)
    assert len(small) <= 502
    assert small.index[0] == frame.index[0] and small.index[-1] == frame.index[-1]


def test_window_labels():
    assert window_label(7) == "7д"
    assert window_label(30) == "1м"
    assert window_label(90) == "3м"
    assert window_label(365) == "1р"
    assert window_label(730) == "2р"
    assert window_label(0.5) == "12год"


def test_unknown_interval_and_smoothing_are_rejected():
    for bad in ("3h", "1w", ""):
        try:
            Timeframe(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for interval {bad!r}")
    try:
        smooth_prices(growth_frame(40, X=0.001), "HULL", 7, INTERVAL)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for an unknown smoothing method")


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            passed += 1
            print(f"ok  {name}")
    print(f"\n{passed} passed")
