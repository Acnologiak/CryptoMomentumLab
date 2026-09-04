"""The simulation's read-only view of prices and signals, as plain numpy.

Per-step `.loc[label]` scalar access is the dominant cost of a bar-by-bar
loop — pandas label lookups are not cheap when you do them tens of thousands
of times. Positional numpy indexing is roughly 10-50x faster, so the engines
work on arrays and only touch pandas when they record a row.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from .signals import SignalFrames


@dataclass
class MarketView:
    index: pd.DatetimeIndex
    columns: list[str]
    alts: list[str]
    base: str
    prices: np.ndarray          # (bars, coins)
    diff: np.ndarray            # (bars, alts) = momentum[alt] - momentum[base]
    alt_momentum: np.ndarray    # (bars, alts)
    column_pos: dict[str, int] = field(init=False)
    base_pos: int = field(init=False)
    alt_pos: list[int] = field(init=False)

    def __post_init__(self) -> None:
        self.column_pos = {name: i for i, name in enumerate(self.columns)}
        self.base_pos = self.column_pos[self.base]
        self.alt_pos = [self.column_pos[a] for a in self.alts]

    @classmethod
    def build(
        cls,
        prices: pd.DataFrame,
        signals: SignalFrames,
        base: str,
        alts: Sequence[str],
    ) -> "MarketView":
        columns = list(prices.columns)
        alts = list(alts)
        return cls(
            index=prices.index,
            columns=columns,
            alts=alts,
            base=base,
            prices=prices[columns].to_numpy(dtype=float),
            diff=signals.diff[alts].to_numpy(dtype=float),
            alt_momentum=signals.momentum[alts].to_numpy(dtype=float),
        )

    @property
    def n_bars(self) -> int:
        return len(self.index)

    @property
    def n_alts(self) -> int:
        return len(self.alts)

    @property
    def n_columns(self) -> int:
        return len(self.columns)

    def time_at(self, pos: int) -> pd.Timestamp:
        return self.index[pos]

    def price_row(self, pos: int) -> np.ndarray:
        return self.prices[pos]

    def checkpoints(self, step: int) -> list[int]:
        """Bar positions the strategy acts on, always including the last bar."""
        positions = list(range(0, self.n_bars, max(1, step)))
        if positions[-1] != self.n_bars - 1:
            positions.append(self.n_bars - 1)
        return positions
