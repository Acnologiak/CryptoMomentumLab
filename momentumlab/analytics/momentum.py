"""Annualized momentum over rolling look-back windows.

Momentum for a window of W days is the growth of the (optionally smoothed)
price between t-W and t, rescaled to a full year:

    compound (CAGR):  (P_t / P_{t-W}) ** (365 / W) - 1
    log (continuous): ln(P_t / P_{t-W}) * 365 / W

Both come out in % per year, so a 30-day window and a 365-day window can be
read off the same axis. The log mode is the default: compounding a +6% week
into a year gives numbers like +285 000% p.a., which makes the chart useless.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from ..timeframe import DAYS_PER_YEAR, Timeframe, window_label
from .smoothing import Smoothing, smooth_prices


class MomentumMode(str, Enum):
    COMPOUND = "compound"
    LOG = "log"

    def __str__(self) -> str:
        return self.value


MOMENTUM_MODES = tuple(m.value for m in MomentumMode)


@dataclass
class MomentumPanel:
    """The same basket of coins measured on several windows at once."""

    curves: dict[float, pd.DataFrame]

    @property
    def windows(self) -> list[float]:
        return sorted(self.curves)

    def __getitem__(self, window_days: float) -> pd.DataFrame:
        return self.curves[float(window_days)]

    def __contains__(self, window_days: float) -> bool:
        return float(window_days) in self.curves

    def __len__(self) -> int:
        return len(self.curves)

    def before(self, moment) -> "MomentumPanel":
        """A copy holding only the rows strictly before `moment`."""
        return MomentumPanel({w: df.loc[df.index < moment] for w, df in self.curves.items()})

    def snapshot(self, as_of=None) -> pd.DataFrame:
        """Coin x window matrix of the most recent momentum value (%, p.a.)."""
        columns = {}
        for window in self.windows:
            frame = self.curves[window]
            if as_of is not None:
                frame = frame.loc[frame.index <= as_of]
            frame = frame.dropna(how="all")
            columns[window_label(window)] = (
                frame.iloc[-1] if not frame.empty else pd.Series(dtype=float)
            )
        snapshot = pd.DataFrame(columns)
        snapshot.index.name = "coin"
        return snapshot


class MomentumCalculator:
    """Smoothing + annualization bundled with the settings they were run at.

    The same instance is reused by the dashboard and by the backtester, which
    is how the simulation is guaranteed to see exactly the curves that are
    drawn on screen.
    """

    def __init__(
        self,
        timeframe: Timeframe | str = "4h",
        smoothing: str | Smoothing = Smoothing.EMA,
        smoothing_days: float = 7.0,
        mode: str | MomentumMode = MomentumMode.LOG,
    ):
        self.timeframe = timeframe if isinstance(timeframe, Timeframe) else Timeframe(timeframe)
        self.smoothing = Smoothing(smoothing)
        self.smoothing_days = smoothing_days
        self.mode = MomentumMode(mode)

    def smooth(self, prices: pd.DataFrame) -> pd.DataFrame:
        return smooth_prices(prices, self.smoothing, self.smoothing_days, self.timeframe)

    def curve(self, series, window_days: float):
        """Annualized growth over a rolling look-back, in % p.a.

        `series` may be a Series or a whole DataFrame; smoothing is assumed to
        have been applied already (see `panel` for the usual entry point).
        """
        if window_days <= 0:
            raise ValueError("window_days must be positive")
        shift = self.timeframe.bars(window_days)
        scale = DAYS_PER_YEAR / float(window_days)
        ratio = series / series.shift(shift)
        ratio = ratio.where(ratio > 0)  # guard against zero / garbage prints
        if self.mode is MomentumMode.COMPOUND:
            out = np.power(ratio, scale) - 1.0
        else:
            out = np.log(ratio) * scale
        return out * 100.0

    def panel(self, prices: pd.DataFrame, windows_days) -> MomentumPanel:
        """Smooth once, then measure every requested window off that series."""
        smoothed = self.smooth(prices)
        return MomentumPanel({
            float(w): self.curve(smoothed, w) for w in sorted(windows_days)
        })
