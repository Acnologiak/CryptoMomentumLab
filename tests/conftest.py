"""Shared synthetic-price builders. No network is touched by any test."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

INTERVAL = "4h"
BARS_PER_DAY = 6


def bar_index(days: int) -> pd.DatetimeIndex:
    return pd.date_range("2021-01-01", periods=days * BARS_PER_DAY, freq="4h", tz="UTC")


def growth_series(daily_rate: float, days: int, start_price: float = 100.0) -> pd.Series:
    """A price series compounding by exactly `daily_rate` every day."""
    idx = bar_index(days)
    per_bar = (1 + daily_rate) ** (1 / BARS_PER_DAY)
    return pd.Series(start_price * per_bar ** np.arange(len(idx)), index=idx)


def growth_frame(days: int = 400, **daily_rates: float) -> pd.DataFrame:
    return pd.DataFrame({name: growth_series(rate, days)
                         for name, rate in daily_rates.items()})


def rollover_frame(days: int = 240, run_share: float = 0.55, run_rate: float = 0.004,
                   fade_rate: float = 0.0001, base_rate: float = 0.0002) -> pd.DataFrame:
    """BTC drifts steadily; ALT runs hard, then decays to a sub-BTC pace so its
    edge over BTC slides linearly through the exit bar and on into negative."""
    idx = bar_index(days)
    n = len(idx)
    cut = int(n * run_share)
    run = 100.0 * ((1 + run_rate) ** np.arange(cut))
    fade = run[-1] * ((1 + fade_rate) ** np.arange(n - cut))
    return pd.DataFrame({
        "BTC": pd.Series(100.0 * ((1 + base_rate) ** np.arange(n)), index=idx),
        "ALT": pd.Series(np.concatenate([run, fade]), index=idx),
    })
