"""Scoring a finished run: headline stats, drawdown, early-exit hindsight."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..timeframe import DAYS_PER_YEAR
from .results import BacktestResult

JUSTIFIED = "виправданий"
PREMATURE = "передчасний"


def drawdown(equity: pd.Series) -> pd.Series:
    """Distance below the running peak, as a negative fraction."""
    return equity / equity.cummax() - 1.0


class PerformanceReport:
    """Everything the dashboard shows about one `BacktestResult`."""

    def __init__(self, result: BacktestResult):
        self.result = result

    # ------------------------------------------------------------------
    def summary(self) -> pd.DataFrame:
        """Strategy vs. buy & hold, side by side."""
        rows = {
            "Стратегія": self._stats(self.result.equity, self.result.fees_paid,
                                     len(self.result.trades)),
            f"Утримання {self.result.base}": self._stats(self.result.hold_equity, 0.0, 0),
        }
        return pd.DataFrame(rows).T

    @staticmethod
    def _stats(equity: pd.Series, fees: float, n_trades: int) -> dict:
        equity = equity.dropna()
        if len(equity) < 2:
            return {}
        days = (equity.index[-1] - equity.index[0]).total_seconds() / 86400
        growth = equity.iloc[-1] / equity.iloc[0]
        cagr = growth ** (DAYS_PER_YEAR / days) - 1.0 if days > 0 else np.nan
        return {
            "дохідність, %": (growth - 1.0) * 100,
            "CAGR, %": cagr * 100,
            "макс. просадка, %": drawdown(equity).min() * 100,
            "угод": n_trades,
            "комісій, % від капіталу": fees * 100,
        }

    def drawdown(self) -> pd.Series:
        return drawdown(self.result.equity)

    def coin_activity(self) -> pd.Series:
        """Share of recorded bars each alt held a slot (%)."""
        if self.result.active.empty:
            return pd.Series(dtype=float)
        return self.result.active.mean().mul(100).sort_values(ascending=False)

    # ------------------------------------------------------------------
    def early_exits(self) -> pd.DataFrame:
        """Score every forced early exit with the benefit of hindsight.

        For each early exit, look forward over the horizon (the forecast
        horizon, or the momentum window when only the fast veto is on) at the
        *slow* signal the ordinary rule watches. If the slow edge would have
        dropped to `exit_edge` (or the alt's own slow momentum turned negative)
        within that horizon, the early exit merely front-ran an exit that was
        coming anyway; otherwise the edge recovered and the exit was premature.
        """
        exits = self.result.early_exits
        if exits.empty:
            return exits.assign(verdict=pd.Series(dtype=object),
                                slow_exit_after_days=pd.Series(dtype=float))

        config = self.result.config
        threshold = config.exit_edge
        horizon = pd.Timedelta(
            days=float(config.forecast_horizon_days or config.window_days))

        scored = []
        for _, row in exits.iterrows():
            moment, coin = row["time"], row["coin"]
            edge = self.result.diff[coin].loc[moment:moment + horizon].iloc[1:]
            own = self.result.momentum[coin].loc[moment:moment + horizon].iloc[1:]
            caught_up = (edge <= threshold) | (own <= 0)
            after = (float((caught_up.idxmax() - moment).total_seconds() / 86400)
                     if caught_up.any() else float("nan"))
            scored.append({**row.to_dict(),
                           "verdict": JUSTIFIED if caught_up.any() else PREMATURE,
                           "slow_exit_after_days": after})
        return pd.DataFrame(scored)
