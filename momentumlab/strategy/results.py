"""What a finished backtest hands back."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .config import StrategyConfig

TRADE_COLUMNS = ["time", "coin", "side", "value", "fee"]
EARLY_EXIT_COLUMNS = ["time", "coin", "trigger", "diff", "predicted_days"]


@dataclass
class BacktestResult:
    equity: pd.Series          # strategy value, indexed like prices, starts at 1.0
    hold_equity: pd.Series     # 100% base-asset buy & hold over the same span
    weights: pd.DataFrame      # portfolio weights at each recorded bar
    active: pd.DataFrame       # bool: did each alt hold a slot at that bar
    trades: pd.DataFrame       # one row per executed leg
    momentum: pd.DataFrame     # momentum used for decisions (simulated span only)
    diff: pd.DataFrame         # momentum[alt] - momentum[base]
    early_exits: pd.DataFrame  # forced early returns to the base asset
    config: StrategyConfig = field(repr=False)

    @property
    def base(self) -> str:
        return self.config.base

    @property
    def alts(self) -> list[str]:
        return list(self.active.columns)

    @property
    def fees_paid(self) -> float:
        return float(self.trades["fee"].sum()) if not self.trades.empty else 0.0

    @property
    def excess_over_base(self) -> pd.Series:
        """Strategy capital divided by buy & hold capital.

        Removes the part of the move both share, leaving only what the
        rotation itself contributed. A flat line at 1.0 means "strategy = BTC".
        """
        ratio = self.equity.dropna() / self.hold_equity.dropna()
        return ratio.dropna()

    @property
    def excess_multiple(self) -> float:
        ratio = self.excess_over_base
        return float(ratio.iloc[-1]) if len(ratio) else float("nan")
