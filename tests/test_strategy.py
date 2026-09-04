"""Synthetic-data checks for the momentum-rotation engines.

Run:  python -m pytest tests -q     (or simply: python tests/test_strategy.py)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from conftest import INTERVAL, bar_index, growth_frame, growth_series, rollover_frame

from momentumlab.strategy import (
    JUSTIFIED,
    PREMATURE,
    PerformanceReport,
    Sizing,
    StrategyConfig,
    run_backtest,
)
from momentumlab.strategy.portfolio import BUY, SELL


def simulate(prices, **kwargs):
    return run_backtest(prices, INTERVAL, StrategyConfig(**kwargs))


def legs(result, coin, side):
    trades = result.trades
    return trades[(trades["coin"] == coin) & (trades["side"] == side)]


def expect_value_error(message, factory):
    try:
        factory()
    except ValueError:
        return
    raise AssertionError(message)


# --------------------------------------------------------------------------
# the core hysteresis
# --------------------------------------------------------------------------
def test_strong_alt_is_bought_and_beats_hold():
    # ALT compounds ~30%/yr faster than BTC -> should get picked up and win.
    prices = growth_frame(BTC=0.0003, ALT=0.0012)
    result = simulate(prices, window_days=14, entry_edge=10, exit_edge=3, max_active=1,
                      rebalance_days=1, fee_bps=5, smoothing="EMA", smoothing_days=3)
    assert result.active["ALT"].sum() > 0, "ALT never got a slot despite outperforming"
    assert result.equity.iloc[-1] > result.hold_equity.iloc[-1]


def test_weak_alt_never_bought_so_strategy_matches_hold_exactly():
    # ALT is flat while BTC grows -> never triggers, so no fees are ever paid.
    prices = growth_frame(BTC=0.001, ALT=0.0)
    result = simulate(prices, window_days=14, entry_edge=10, exit_edge=3,
                      max_active=1, rebalance_days=1, fee_bps=5)
    assert result.active["ALT"].sum() == 0
    assert result.trades.empty
    np.testing.assert_allclose(result.equity.values, result.hold_equity.values, rtol=1e-9)


def test_negative_momentum_alt_is_never_bought_even_if_it_beats_a_crashing_btc():
    # BTC crashes hard, ALT merely drifts down slower -> ALT wins on the diff,
    # but its own momentum is negative, so the entry rule must block it.
    prices = growth_frame(BTC=-0.01, ALT=-0.001)
    result = simulate(prices, window_days=14, entry_edge=5, exit_edge=1,
                      max_active=1, rebalance_days=1)
    assert result.active["ALT"].sum() == 0


def test_rotation_respects_max_active_slots():
    prices = growth_frame(BTC=0.0002, A=0.0015, B=0.0014, C=0.0013)
    result = simulate(prices, window_days=10, entry_edge=8, exit_edge=2, max_active=2,
                      rebalance_days=1, smoothing="none")
    assert result.active[["A", "B", "C"]].sum(axis=1).max() <= 2
    assert (result.weights[["A", "B", "C"]].max(axis=1) <= 0.5 + 1e-9).all()


def test_fees_reduce_equity_relative_to_a_zero_fee_run():
    prices = growth_frame(BTC=0.0003, ALT=0.0012)
    common = dict(window_days=14, entry_edge=10, exit_edge=3, max_active=1,
                  rebalance_days=1)
    cheap = simulate(prices, **common, fee_bps=0)
    pricey = simulate(prices, **common, fee_bps=200)
    assert pricey.equity.iloc[-1] < cheap.equity.iloc[-1]
    assert not pricey.trades.empty
    assert pricey.trades["fee"].sum() > 0


def test_start_end_slices_simulation_but_keeps_full_warmup():
    prices = growth_frame(BTC=0.0003, ALT=0.0012)
    config = StrategyConfig(window_days=14, entry_edge=10, exit_edge=3, max_active=1)
    late_start = prices.index[int(len(prices) * 0.6)]
    result = run_backtest(prices, INTERVAL, config, start=late_start)

    assert result.equity.index[0] >= late_start
    # momentum is already warmed up at the sliced start: no leading NaN gap
    assert result.momentum.iloc[0].notna().all()


def test_late_listed_coin_does_not_poison_equity_with_nan():
    # LATE has no history for its first 100 days, like a coin that listed well
    # after BTC. It cannot qualify for a slot during that gap (no momentum yet),
    # so the portfolio sits in 100% BTC and equity must track BTC exactly --
    # not go NaN because 0 qty * NaN price poisoned the sum.
    btc = growth_series(0.0006, 400)
    late = growth_series(0.002, 300)
    late = pd.concat([pd.Series(np.nan, index=btc.index[:len(btc) - len(late)]), late])
    late.index = btc.index
    prices = pd.DataFrame({"BTC": btc, "LATE": late})

    for sizing in Sizing:
        result = simulate(prices, window_days=14, entry_edge=10, exit_edge=3,
                          max_active=1, fee_bps=5, sizing=sizing, smoothing="none")
        before_listing = result.equity.iloc[:100 * 6]
        assert before_listing.notna().all(), f"equity went NaN before LATE listed ({sizing})"
        expected = prices["BTC"].iloc[:100 * 6] / prices["BTC"].iloc[0]
        np.testing.assert_allclose(before_listing.values, expected.values, rtol=1e-9)
        assert result.weights.notna().all().all()


# --------------------------------------------------------------------------
# incremental sizing
# --------------------------------------------------------------------------
def test_incremental_simultaneous_entries_split_btc_33_50_100():
    # A > B > C all cross the entry bar on the very same bar (constant growth
    # rates are valid from the first warmed-up bar onward), so the sequential
    # 1/3 -> half of the rest -> all of the rest split must land each alt at
    # roughly an even third, funded entirely out of BTC.
    prices = growth_frame(BTC=0.0002, A=0.0020, B=0.0019, C=0.0018)
    result = simulate(prices, window_days=10, entry_edge=10, exit_edge=2, max_active=3,
                      fee_bps=0, sizing=Sizing.INCREMENTAL, smoothing="none")

    final = result.weights.iloc[-1]
    for coin in ("A", "B", "C"):
        assert abs(final[coin] - 1 / 3) < 0.02, (coin, final[coin])
    assert final["BTC"] < 1e-6

    buys = result.trades[result.trades["side"] == BUY]
    assert set(buys["coin"]) == {"A", "B", "C"}
    assert buys["time"].nunique() == 1  # all three fired on the same bar


def test_incremental_existing_position_is_never_resized():
    # A enters alone while B/C stay flat; nothing later may touch its holding,
    # so A appears in exactly one trade row for the whole run.
    prices = growth_frame(BTC=0.0002, A=0.0020, B=0.0, C=0.0)
    result = simulate(prices, window_days=10, entry_edge=10, exit_edge=2, max_active=3,
                      fee_bps=5, sizing=Sizing.INCREMENTAL, smoothing="none")

    a_trades = result.trades[result.trades["coin"] == "A"]
    assert len(a_trades) == 1 and a_trades.iloc[0]["side"] == BUY
    assert result.active["A"].sum() > 0
    assert result.active["B"].sum() == 0 and result.active["C"].sum() == 0


def test_incremental_exit_sells_the_full_position_back_to_btc():
    idx = bar_index(200)
    btc = pd.Series(100.0 * (1.0003 ** np.arange(len(idx))), index=idx)
    # ALT beats BTC for the first half, then collapses -> momentum goes negative.
    half = len(idx) // 2
    up = 100.0 * (1.003 ** np.arange(half))
    down = up[-1] * (0.995 ** np.arange(len(idx) - half))
    prices = pd.DataFrame({"BTC": btc,
                           "ALT": pd.Series(np.concatenate([up, down]), index=idx)})

    result = simulate(prices, window_days=10, entry_edge=10, exit_edge=2, max_active=1,
                      fee_bps=0, sizing=Sizing.INCREMENTAL, smoothing="none")

    sells = legs(result, "ALT", SELL)
    assert not sells.empty, "ALT never got sold back to BTC after collapsing"
    after_exit = result.weights.loc[sells.iloc[0]["time"]]
    assert after_exit["ALT"] < 1e-9
    assert after_exit["BTC"] > 0.99


def test_incremental_ignores_rebalance_days_and_checks_every_bar():
    prices = growth_frame(BTC=0.0003, ALT=0.0012)
    result = simulate(prices, window_days=14, entry_edge=10, exit_edge=3, max_active=1,
                      rebalance_days=30, sizing=Sizing.INCREMENTAL)
    assert len(result.weights) == len(prices)


# --------------------------------------------------------------------------
# slot rotation
# --------------------------------------------------------------------------
def _rotation_prices(with_flat_filler: bool = False) -> pd.DataFrame:
    """A holds the only slot with a moderate edge; B does nothing, then rockets."""
    idx = bar_index(200)
    n = len(idx)
    half = n // 2
    frame = {
        "BTC": pd.Series(100.0 * (1.0001 ** np.arange(n)), index=idx),
        "A": pd.Series(100.0 * (1.0006 ** np.arange(n)), index=idx),
        "B": pd.Series(np.concatenate([np.full(half, 100.0),
                                       100.0 * (1.006 ** np.arange(n - half))]), index=idx),
    }
    if with_flat_filler:
        frame["C"] = pd.Series(np.full(n, 100.0), index=idx)  # never a contender
    return pd.DataFrame(frame)


def test_rotation_evicts_a_weak_slot_for_a_much_stronger_outsider():
    result = simulate(_rotation_prices(with_flat_filler=True), window_days=10,
                      entry_edge=10, exit_edge=3, max_active=1, fee_bps=0,
                      sizing=Sizing.INCREMENTAL, smoothing="none", rotation_margin=15.0)

    assert result.active["A"].sum() > 0, "A should have held the only slot early on"
    assert result.active["B"].sum() > 0, "B should have rotated in once it took off"
    assert not legs(result, "B", BUY).empty
    assert not legs(result, "A", SELL).empty
    # The rotation is a single BTC-routed swap, so with one slot the two coins
    # must never be held at the same time.
    assert (result.active[["A", "B"]].sum(axis=1) <= 1).all()


def test_rotation_disabled_by_default_keeps_a_held_slot():
    # Same setup, rotation_margin left at None -> B must never get a slot while
    # A still individually clears the (lenient) exit bar, however far B is ahead.
    result = simulate(_rotation_prices(), window_days=10, entry_edge=10, exit_edge=3,
                      max_active=1, fee_bps=0, sizing=Sizing.INCREMENTAL,
                      smoothing="none")
    assert result.active["A"].sum() > 0
    assert result.active["B"].sum() == 0, "without a margin, B must never displace A"


# --------------------------------------------------------------------------
# early return to BTC
# --------------------------------------------------------------------------
def test_predictive_early_exit_returns_to_btc_before_the_slow_rule():
    prices = rollover_frame()
    common = dict(window_days=15, entry_edge=12, exit_edge=2, max_active=1,
                  fee_bps=0, sizing=Sizing.INCREMENTAL, smoothing="none")
    off = simulate(prices, **common)
    on = simulate(prices, **common, forecast_horizon_days=8, slope_window_days=5)

    assert legs(off, "ALT", BUY).iloc[0]["time"] == legs(on, "ALT", BUY).iloc[0]["time"], \
        "an early exit must never move the entry"
    assert legs(on, "ALT", SELL).iloc[0]["time"] < legs(off, "ALT", SELL).iloc[0]["time"]
    assert on.active["ALT"].sum() < off.active["ALT"].sum()
    assert not on.early_exits.empty
    assert (on.early_exits["trigger"] == "прогноз").all()
    assert (on.early_exits["predicted_days"] > 0).all()
    # the slow edge really does fall through the bar afterwards -> all justified
    assert (PerformanceReport(on).early_exits()["verdict"] == JUSTIFIED).all()


def test_predictive_early_exit_also_fires_under_target_weight_sizing():
    result = simulate(rollover_frame(), window_days=15, entry_edge=12, exit_edge=2,
                      max_active=1, fee_bps=0, sizing=Sizing.TARGET_WEIGHT,
                      smoothing="none", forecast_horizon_days=8, slope_window_days=5)
    assert not result.early_exits.empty
    assert (result.early_exits["trigger"] == "прогноз").all()
    verdicts = set(PerformanceReport(result).early_exits()["verdict"])
    assert verdicts <= {JUSTIFIED, PREMATURE}


def test_predictive_early_exit_is_inert_on_a_steady_winner():
    # Constant growth rates -> the diff is flat -> slope ~0 -> nothing to predict.
    prices = growth_frame(BTC=0.0003, ALT=0.0012)
    common = dict(window_days=14, entry_edge=10, exit_edge=3, max_active=1,
                  fee_bps=0, sizing=Sizing.INCREMENTAL)
    off = simulate(prices, **common)
    on = simulate(prices, **common, forecast_horizon_days=10, slope_window_days=5)

    assert on.early_exits.empty
    np.testing.assert_allclose(on.equity.values, off.equity.values, rtol=1e-9)


def test_fast_window_veto_exits_while_the_slow_window_still_holds():
    idx = bar_index(260)
    n = len(idx)
    run, dump_len = int(n * 0.5), int(n * 0.08)
    up = 100.0 * (1.004 ** np.arange(run))
    dump = up[-1] * (0.997 ** np.arange(dump_len))            # short, sharp drop
    recovery = dump[-1] * (1.0002 ** np.arange(n - run - dump_len))
    prices = pd.DataFrame({
        "BTC": pd.Series(100.0 * (1.0002 ** np.arange(n)), index=idx),
        "ALT": pd.Series(np.concatenate([up, dump, recovery]), index=idx),
    })

    common = dict(window_days=45, entry_edge=12, exit_edge=2, max_active=1,
                  fee_bps=0, sizing=Sizing.INCREMENTAL, smoothing="none")
    off = simulate(prices, **common)
    on = simulate(prices, **common, fast_window_days=7)

    on_sells = legs(on, "ALT", SELL)
    assert not on_sells.empty
    assert not on.early_exits.empty
    assert (on.early_exits["trigger"] == "швидкий момент").all()

    fired_at = on.early_exits.iloc[0]["time"]
    assert on.diff.loc[fired_at, "ALT"] > common["exit_edge"], \
        "the veto should fire while the slow edge is still above the stay bar"
    assert on.active["ALT"].sum() < off.active["ALT"].sum()

    off_sells = legs(off, "ALT", SELL)
    if not off_sells.empty:
        assert on_sells.iloc[0]["time"] <= off_sells.iloc[0]["time"]


def test_early_exit_report_is_shaped_even_when_the_feature_is_off():
    prices = growth_frame(BTC=0.0003, ALT=0.0012)
    result = simulate(prices, window_days=14, entry_edge=10, exit_edge=3, max_active=1)
    assert result.early_exits.empty
    report = PerformanceReport(result).early_exits()
    assert report.empty
    assert {"verdict", "slow_exit_after_days"} <= set(report.columns)


# --------------------------------------------------------------------------
# reporting and validation
# --------------------------------------------------------------------------
def test_summary_and_activity_shapes():
    prices = growth_frame(BTC=0.0003, ALT=0.0012)
    result = simulate(prices, window_days=14, entry_edge=10, exit_edge=3, max_active=1)
    report = PerformanceReport(result)

    summary = report.summary()
    assert set(summary.index) == {"Стратегія", "Утримання BTC"}
    assert {"дохідність, %", "CAGR, %", "макс. просадка, %", "угод"} <= set(summary.columns)
    assert list(report.coin_activity().index) == ["ALT"]
    assert report.drawdown().max() <= 1e-12


def test_excess_over_base_is_the_capital_ratio():
    prices = growth_frame(BTC=0.0003, ALT=0.0012)
    result = simulate(prices, window_days=14, entry_edge=10, exit_edge=3, max_active=1)
    excess = result.excess_over_base
    np.testing.assert_allclose(excess.values,
                               (result.equity / result.hold_equity).values, rtol=1e-12)
    assert result.excess_multiple > 1.0


def test_config_validation():
    expect_value_error("exit_edge above entry_edge must be rejected",
                       lambda: StrategyConfig(entry_edge=5, exit_edge=10))
    expect_value_error("max_active below 1 must be rejected",
                       lambda: StrategyConfig(max_active=0))
    expect_value_error("a negative rotation margin must be rejected",
                       lambda: StrategyConfig(rotation_margin=-1.0))
    for bad in (dict(forecast_horizon_days=0.0), dict(forecast_horizon_days=-3.0),
                dict(slope_window_days=0.0), dict(fast_window_days=0.0),
                dict(fast_window_days=-1.0)):
        expect_value_error(f"expected ValueError for {bad}",
                           lambda kw=bad: StrategyConfig(**kw))


def test_missing_base_asset_is_rejected():
    prices = growth_frame(ETH=0.001, ALT=0.001)
    expect_value_error("a missing base asset must be rejected",
                       lambda: simulate(prices, base="BTC"))


def test_basket_without_alts_is_rejected():
    prices = growth_frame(BTC=0.001)
    expect_value_error("a base-only basket must be rejected",
                       lambda: simulate(prices, base="BTC"))


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            passed += 1
            print(f"ok  {name}")
    print(f"\n{passed} passed")
