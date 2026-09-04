"""Price smoothers, all configured in days rather than bars.

Smoothing runs *before* the momentum calculation, so it is the main knob for
trading noise against lag. Because every method takes its period in days, the
setting keeps its meaning when the candle interval changes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

import pandas as pd

from ..timeframe import Timeframe


class Smoothing(str, Enum):
    NONE = "none"
    SMA = "SMA"
    EMA = "EMA"
    ZLEMA = "ZLEMA"

    def __str__(self) -> str:
        return self.value


class Smoother(ABC):
    """Turns a price frame into a smoothed price frame of the same shape."""

    method: Smoothing

    @abstractmethod
    def apply(self, prices: pd.DataFrame, span: int) -> pd.DataFrame:
        """`span` is the smoothing period already converted to bars."""


class PassThrough(Smoother):
    method = Smoothing.NONE

    def apply(self, prices, span):
        return prices


class SimpleMovingAverage(Smoother):
    method = Smoothing.SMA

    def apply(self, prices, span):
        return prices.rolling(span, min_periods=span).mean()


class ExponentialMovingAverage(Smoother):
    method = Smoothing.EMA

    def apply(self, prices, span):
        return prices.ewm(span=span, adjust=False, min_periods=span).mean()


class ZeroLagEMA(Smoother):
    """A plain EMA fed a lag-compensated input.

    Each sample is pushed forward by the momentum it has built up over the
    EMA's own group delay (`lag = (span - 1) // 2` bars) before smoothing:
    `EMA(price + (price - price[-lag]))`. On a locally linear trend this
    cancels the EMA's lag almost exactly; on sharp turns it overshoots, so it
    trades a little stability for a faster reaction to reversals.
    """

    method = Smoothing.ZLEMA

    def apply(self, prices, span):
        lag = (span - 1) // 2
        de_lagged = prices + (prices - prices.shift(lag))
        return de_lagged.ewm(span=span, adjust=False, min_periods=span + lag).mean()


_SMOOTHERS: dict[Smoothing, Smoother] = {
    s.method: s for s in (PassThrough(), SimpleMovingAverage(),
                          ExponentialMovingAverage(), ZeroLagEMA())
}

SMOOTHING_METHODS = tuple(m.value for m in Smoothing)


def get_smoother(method: str | Smoothing) -> Smoother:
    try:
        return _SMOOTHERS[Smoothing(method)]
    except ValueError:
        raise ValueError(f"unknown smoothing method {method!r}") from None


def smooth_prices(
    prices: pd.DataFrame,
    method: str | Smoothing = Smoothing.EMA,
    days: float = 7.0,
    timeframe: Timeframe | str = "4h",
) -> pd.DataFrame:
    """Convenience wrapper: pick a smoother and run it over `days` of history."""
    if days <= 0:
        return prices
    smoother = get_smoother(method)
    if smoother.method is Smoothing.NONE:
        return prices
    timeframe = timeframe if isinstance(timeframe, Timeframe) else Timeframe(timeframe)
    return smoother.apply(prices, timeframe.bars(days))
