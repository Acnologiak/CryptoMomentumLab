"""Every knob of the rotation strategy, in one validated object.

The strategy itself is a two-threshold hysteresis around a single number: how
far an alt's annualized momentum sits above BTC's (`diff`). It enters when the
edge clears `entry_edge`, and it stays until the edge falls below `exit_edge`.
Holding between the two bars is what stops the portfolio from flapping every
time the gap wobbles by a point.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..analytics import MomentumMode, Smoothing


class Sizing(str, Enum):
    """How much of the portfolio a freshly opened slot receives."""

    #: Each active alt targets a fixed 1/K share of the *whole* portfolio and
    #: every checkpoint pins it back there, open positions included.
    TARGET_WEIGHT = "target_weight"
    #: Opportunistic swap-when-you-can, no periodic rebalancing at all. An
    #: entry spends `BTC_balance / free_slots`; an exit sells that one position
    #: in full. Existing positions are never resized by later signals.
    INCREMENTAL = "incremental"

    def __str__(self) -> str:
        return self.value


class ExitTrigger(str, Enum):
    """Why an early return to BTC fired (used in the trade journal)."""

    FORECAST = "прогноз"
    FAST_WINDOW = "швидкий момент"

    def __str__(self) -> str:
        return self.value


SIZING_MODES = tuple(s.value for s in Sizing)


@dataclass
class StrategyConfig:
    """Parameters of one backtest run.

    Threshold units are percentage points of annualized momentum, i.e. the
    same units the momentum curves are drawn in.
    """

    base: str = "BTC"
    window_days: float = 30.0
    entry_edge: float = 20.0       # edge over BTC needed to OPEN a slot
    exit_edge: float = 5.0         # edge over BTC needed to KEEP it (<= entry_edge)
    max_active: int = 4            # K simultaneous alt slots
    rebalance_days: float = 1.0    # target_weight only: how often the strategy re-trades
    fee_bps: float = 10.0          # cost per BTC<->alt leg, in bps of notional
    smoothing: Smoothing = Smoothing.EMA
    smoothing_days: float = 7.0
    mode: MomentumMode = MomentumMode.LOG
    sizing: Sizing = Sizing.TARGET_WEIGHT

    #: Incremental only. When set, an outside coin that already clears
    #: `entry_edge` and beats the weakest held slot by more than this margin
    #: evicts it. `None` disables rotation: a held coin is only ever displaced
    #: by its own decline. The margin exists for the same reason `exit_edge`
    #: does — two coins swapping rank every bar would otherwise trade on noise.
    rotation_margin: float | None = None

    # --- early return to BTC; both default off, both only ever pull an exit
    # forward. See `signals.EarlyExitRule` for what they actually compute.
    forecast_horizon_days: float | None = None  # predictive slope crossing horizon
    slope_window_days: float = 5.0              # look-back for the least-squares slope
    fast_window_days: float | None = None       # veto via a shorter momentum window

    def __post_init__(self) -> None:
        self.smoothing = Smoothing(self.smoothing)
        self.mode = MomentumMode(self.mode)
        self.sizing = Sizing(self.sizing)

        if self.exit_edge > self.entry_edge:
            raise ValueError(
                "поріг виходу має бути <= порогу входу "
                "(інакше вхід і вихід суперечать одне одному)"
            )
        if self.max_active < 1:
            raise ValueError("кількість слотів має бути >= 1")
        if self.rotation_margin is not None and self.rotation_margin < 0:
            raise ValueError("поріг ротації має бути >= 0 (або None, щоб вимкнути ротацію)")
        if self.forecast_horizon_days is not None and self.forecast_horizon_days <= 0:
            raise ValueError("горизонт прогнозу має бути > 0 (або None, щоб вимкнути прогноз)")
        if self.slope_window_days <= 0:
            raise ValueError("вікно нахилу має бути > 0")
        if self.fast_window_days is not None and self.fast_window_days <= 0:
            raise ValueError("швидке вікно має бути > 0 (або None, щоб вимкнути вето)")

    @property
    def fee_rate(self) -> float:
        return self.fee_bps / 10_000.0

    @property
    def uses_forecast_exit(self) -> bool:
        return self.forecast_horizon_days is not None

    @property
    def uses_fast_veto(self) -> bool:
        return self.fast_window_days is not None

    @property
    def has_early_exit(self) -> bool:
        return self.uses_forecast_exit or self.uses_fast_veto
