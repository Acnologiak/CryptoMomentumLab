"""The bundle of already-computed state every tab reads from."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..analytics import MomentumPanel
from .charts import ChartFactory
from .settings import AppSettings


@dataclass
class DashboardContext:
    """Prices, momentum and the selected display period, computed once."""

    settings: AppSettings
    prices: pd.DataFrame
    errors: dict[str, str]
    panel: MomentumPanel
    snapshot: pd.DataFrame
    colors: dict[str, str]
    charts: ChartFactory
    low: pd.Timestamp   # inclusive start of the displayed period
    high: pd.Timestamp  # exclusive end

    @property
    def windows(self) -> list[float]:
        return self.panel.windows

    @property
    def coins(self) -> list[str]:
        """Selected coins, in sidebar order, that actually have prices."""
        return [c for c in self.settings.coins if c in self.prices.columns]

    def in_range(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Crop to the displayed period and to the coins the user selected."""
        rows = (frame.index >= self.low) & (frame.index < self.high)
        return frame.loc[rows, [c for c in self.settings.coins if c in frame.columns]]
