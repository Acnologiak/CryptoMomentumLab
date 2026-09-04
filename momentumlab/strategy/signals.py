"""Turning prices into the numbers the rotation rules actually look at.

`SignalBuilder` computes them on the *full* price history, so the momentum
look-back is always warmed up from real data even when the simulation itself
is restricted to a later window. `EarlyExitRule` then reads one bar at a time
during the simulation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from ..analytics import MomentumCalculator
from ..timeframe import Timeframe
from .config import ExitTrigger, StrategyConfig


def rolling_ols_slope(frame: pd.DataFrame, n: int) -> pd.DataFrame:
    """Trailing least-squares slope of each column over an `n`-bar window.

    Units are (column units) per bar. NaN until the first full window, and for
    any window that contains a NaN.

    slope = sum((k - k_mean) * y_k) / sum((k - k_mean)^2) with k = 0..n-1 fixed,
    so it is just a centred-ramp kernel dotted with each trailing window — one
    strided view and a matmul instead of an n-times-longer Python loop.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    k = np.arange(n) - (n - 1) / 2.0
    kernel = k / float(k @ k)
    values = frame.to_numpy(dtype=float)
    out = np.full(values.shape, np.nan)
    if len(values) >= n:
        windows = sliding_window_view(values, n, axis=0)      # (T-n+1, C, n)
        slopes = windows @ kernel
        slopes[np.isnan(windows).any(axis=-1)] = np.nan
        out[n - 1:] = slopes
    return pd.DataFrame(out, index=frame.index, columns=frame.columns)


@dataclass
class SignalFrames:
    """Everything the decision rules read, aligned on the price index."""

    momentum: pd.DataFrame                   # every coin, base included
    diff: pd.DataFrame                       # alts only: momentum[alt] - momentum[base]
    slope: pd.DataFrame | None = None        # trailing OLS slope of `diff`, pp/bar
    fast_diff: pd.DataFrame | None = None    # `diff` recomputed on the fast window
    fast_momentum: pd.DataFrame | None = None

    def restricted_to(self, index: pd.DatetimeIndex) -> "SignalFrames":
        """The same signals, cropped to the simulation window."""
        crop = lambda df: None if df is None else df.loc[index]  # noqa: E731
        return SignalFrames(
            momentum=self.momentum.loc[index],
            diff=self.diff.loc[index],
            slope=crop(self.slope),
            fast_diff=crop(self.fast_diff),
            fast_momentum=crop(self.fast_momentum),
        )


class SignalBuilder:
    """Builds `SignalFrames` for one config over one price history."""

    def __init__(self, config: StrategyConfig, timeframe: Timeframe):
        self.config = config
        self.timeframe = timeframe
        self.calculator = MomentumCalculator(
            timeframe=timeframe,
            smoothing=config.smoothing,
            smoothing_days=config.smoothing_days,
            mode=config.mode,
        )

    def build(self, prices: pd.DataFrame, alts: Sequence[str]) -> SignalFrames:
        config = self.config
        smoothed = self.calculator.smooth(prices)

        momentum = self._momentum_over(smoothed, prices.columns, config.window_days)
        diff = momentum[list(alts)].sub(momentum[config.base], axis=0)

        slope = None
        if config.uses_forecast_exit:
            bars = max(2, self.timeframe.bars(config.slope_window_days))
            slope = rolling_ols_slope(diff, bars)

        fast_diff = fast_momentum = None
        if config.uses_fast_veto:
            fast_momentum = self._momentum_over(
                smoothed, prices.columns, config.fast_window_days)
            fast_diff = fast_momentum[list(alts)].sub(fast_momentum[config.base], axis=0)

        return SignalFrames(momentum=momentum, diff=diff, slope=slope,
                            fast_diff=fast_diff, fast_momentum=fast_momentum)

    def _momentum_over(self, smoothed, columns, window_days) -> pd.DataFrame:
        return pd.DataFrame({
            column: self.calculator.curve(smoothed[column], window_days)
            for column in columns
        })


@dataclass
class EarlyExitVerdict:
    """Per-alt outcome of the early-exit check on a single bar."""

    fired: np.ndarray                 # bool, one entry per alt
    triggers: list[ExitTrigger | None]
    predicted_days: np.ndarray        # only meaningful where `fired`

    def any(self) -> bool:
        return bool(self.fired.any())


class EarlyExitRule:
    """The two optional accelerators that pull a return to BTC forward.

    Both exist to claw back the lag that smoothing plus the momentum look-back
    bake into the ordinary `exit_edge` exit. They only ever get you *out*
    sooner: while the condition holds for a coin it is forced back to BTC and
    also kept from re-entering on that same bar (re-opening a position the rule
    would unwind next bar is just a fee pump), but a clean entry signal is
    never delayed and a position is never held longer than plain hysteresis
    would hold it.

    * Predictive crossing — fit a least-squares line through the last
      `slope_window_days` of the alt's edge over BTC and extrapolate it. If
      `diff` is still above `exit_edge` but that straight line is projected to
      cross it within `forecast_horizon_days`, unwind now instead of waiting
      for the real crossing. It is a derivative term on the exit rule, and it
      only fires while the fitted slope is negative.
    * Fast-window veto — recompute the edge on a shorter momentum window. If
      that faster read has already lost the edge (or the alt's own fast
      momentum has turned negative), unwind even though the slow window still
      clears the stay bar. Less lag, more churn on noise.
    """

    def __init__(
        self,
        config: StrategyConfig,
        timeframe: Timeframe,
        slope: np.ndarray | None = None,
        fast_diff: np.ndarray | None = None,
        fast_momentum: np.ndarray | None = None,
    ):
        self.config = config
        self.bars_per_day = timeframe.bars_per_day
        self.slope = slope
        self.fast_diff = fast_diff
        self.fast_momentum = fast_momentum
        self.horizon_bars = (
            config.forecast_horizon_days * timeframe.bars_per_day
            if config.uses_forecast_exit else 0.0
        )

    @property
    def enabled(self) -> bool:
        forecast_ready = self.horizon_bars > 0 and self.slope is not None
        return forecast_ready or self.fast_diff is not None

    def evaluate(self, pos: int, diff_row: np.ndarray) -> EarlyExitVerdict:
        """Which alts an early-exit condition holds for *right now*.

        Independent of whether they are currently held: callers subtract the
        mask from both the stay set and the entry set.
        """
        n = diff_row.size
        fired = np.zeros(n, dtype=bool)
        triggers: list[ExitTrigger | None] = [None] * n
        predicted_days = np.full(n, np.nan)

        if self.horizon_bars > 0 and self.slope is not None:
            slope_row = self.slope[pos]
            with np.errstate(divide="ignore", invalid="ignore"):
                bars_to_cross = (self.config.exit_edge - diff_row) / slope_row
            hit = (
                (slope_row < 0)
                & (diff_row > self.config.exit_edge)
                & np.isfinite(bars_to_cross)
                & (bars_to_cross > 0)
                & (bars_to_cross <= self.horizon_bars)
            )
            for i in np.where(hit)[0]:
                fired[i] = True
                triggers[i] = ExitTrigger.FORECAST
                predicted_days[i] = bars_to_cross[i] / self.bars_per_day

        if self.fast_diff is not None:
            fast_d, fast_m = self.fast_diff[pos], self.fast_momentum[pos]
            veto = (
                ~fired
                & np.isfinite(fast_d) & np.isfinite(fast_m)
                & ((fast_d <= self.config.exit_edge) | (fast_m <= 0))
            )
            for i in np.where(veto)[0]:
                fired[i] = True
                triggers[i] = ExitTrigger.FAST_WINDOW

        return EarlyExitVerdict(fired, triggers, predicted_days)
