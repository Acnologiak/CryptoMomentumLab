"""Who is in front, and for how long — derived views over one momentum frame."""
from __future__ import annotations

import pandas as pd


class RankingAnalyzer:
    """Rank / leader statistics for a single momentum window."""

    def __init__(self, momentum: pd.DataFrame):
        self.momentum = momentum

    def ranks(self, top_is_first: bool = True) -> pd.DataFrame:
        """Rank every coin at every timestamp (1 = strongest momentum)."""
        return self.momentum.rank(axis=1, ascending=not top_is_first, method="min")

    def leader(self) -> pd.Series:
        """Which coin holds the strongest momentum at each timestamp."""
        valid = self.momentum.dropna(how="all")
        if valid.empty:
            return pd.Series(dtype=object)
        return valid.idxmax(axis=1)

    def leader_shares(self) -> pd.Series:
        """Share of time each coin spent as the momentum leader (%)."""
        leader = self.leader()
        if leader.empty:
            return pd.Series(dtype=float)
        return leader.value_counts(normalize=True).mul(100).sort_values(ascending=False)
