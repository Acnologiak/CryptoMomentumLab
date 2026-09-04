"""Bar-length arithmetic shared by every layer of the project.

Everything user-facing in this app is configured in *days* — momentum
windows, smoothing periods, rebalance cadence — so switching the candle
interval never silently changes the meaning of a setting. `Timeframe` is the
single place where days are translated into bars and back.
"""
from __future__ import annotations

from dataclasses import dataclass

DAYS_PER_YEAR = 365.0
MINUTES_PER_DAY = 1440.0

# Candle intervals exposed in the UI -> their length in minutes.
INTERVAL_MINUTES: dict[str, int] = {
    "1h": 60,
    "2h": 120,
    "4h": 240,
    "6h": 360,
    "12h": 720,
    "1d": 1440,
}

SUPPORTED_INTERVALS = tuple(INTERVAL_MINUTES)


@dataclass(frozen=True)
class Timeframe:
    """One candle interval, plus the conversions that depend on it."""

    interval: str = "4h"

    def __post_init__(self) -> None:
        if self.interval not in INTERVAL_MINUTES:
            raise ValueError(
                f"unsupported interval {self.interval!r}; "
                f"expected one of {', '.join(SUPPORTED_INTERVALS)}"
            )

    @property
    def minutes(self) -> int:
        return INTERVAL_MINUTES[self.interval]

    @property
    def bars_per_day(self) -> float:
        return MINUTES_PER_DAY / self.minutes

    @property
    def pandas_freq(self) -> str:
        """Frequency alias for `DataFrame.resample` / `date_range`."""
        return f"{self.minutes}min"

    @property
    def milliseconds(self) -> int:
        return self.minutes * 60_000

    def bars(self, days: float) -> int:
        """How many bars cover `days` on this interval (never fewer than 1)."""
        return max(1, int(round(days * self.bars_per_day)))

    def days(self, bars: float) -> float:
        return bars / self.bars_per_day

    def __str__(self) -> str:
        return self.interval


def window_label(days: float) -> str:
    """Short Ukrainian label for a look-back window: 7д, 1м, 3м, 1р, 12год."""
    days = float(days)
    if days >= 365 and days % 365 == 0:
        years = int(days // 365)
        return f"{years}р" if years > 1 else "1р"
    if days >= 30 and days % 30 == 0:
        return f"{int(days // 30)}м"
    if days >= 1:
        return f"{int(days)}д" if days.is_integer() else f"{days:g}д"
    return f"{days * 24:g}год"
