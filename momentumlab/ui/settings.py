"""The sidebar's answer, as one object the rest of the app can read."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..analytics import MomentumCalculator, MomentumMode, Smoothing
from ..config import (
    DEFAULT_EXCHANGE,
    DEFAULT_HISTORY_YEARS,
    DEFAULT_INTERVAL,
    DEFAULT_MAX_PLOT_POINTS,
    DEFAULT_MOMENTUM_MODE,
    DEFAULT_QUOTE,
    DEFAULT_SMOOTHING,
    DEFAULT_SMOOTHING_DAYS,
)
from ..timeframe import Timeframe


@dataclass
class AppSettings:
    """Everything the user picked in the sidebar for this rerun."""

    exchange: str = DEFAULT_EXCHANGE
    quote: str = DEFAULT_QUOTE
    interval: str = DEFAULT_INTERVAL
    years: float = DEFAULT_HISTORY_YEARS
    coins: list[str] = field(default_factory=list)

    smoothing: Smoothing = Smoothing(DEFAULT_SMOOTHING)
    smoothing_days: float = DEFAULT_SMOOTHING_DAYS
    windows: list[float] = field(default_factory=list)
    mode: MomentumMode = MomentumMode(DEFAULT_MOMENTUM_MODE)

    clip_axis: bool = True
    clip_limit: int = 1000
    max_points: int = DEFAULT_MAX_PLOT_POINTS
    show_zero: bool = True

    #: bumped by the "refresh" button to invalidate Streamlit's data cache
    cache_bust: int = 0

    @property
    def timeframe(self) -> Timeframe:
        return Timeframe(self.interval)

    @property
    def history_days(self) -> int:
        return int(self.years * 365.25)

    @property
    def start_date(self) -> str:
        first = pd.Timestamp.now("UTC").normalize() - pd.Timedelta(days=self.history_days)
        return first.strftime("%Y-%m-%d")

    @property
    def mode_caption(self) -> str:
        return "CAGR" if self.mode is MomentumMode.COMPOUND else "безперервне нарахування"

    def calculator(self) -> MomentumCalculator:
        """The momentum engine configured exactly as the sidebar says."""
        return MomentumCalculator(
            timeframe=self.timeframe,
            smoothing=self.smoothing,
            smoothing_days=self.smoothing_days,
            mode=self.mode,
        )
