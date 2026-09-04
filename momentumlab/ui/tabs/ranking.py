"""Tab 2 — who leads right now, and who has been leading."""
from __future__ import annotations

import streamlit as st

from ...analytics import RankingAnalyzer
from ...timeframe import window_label
from ..charts import styled_snapshot
from ..context import DashboardContext
from .base import DashboardTab


class RankingTab(DashboardTab):
    title = "🏆 Рейтинг"

    def render(self, ctx: DashboardContext) -> None:
        self._current(ctx)
        self._history(ctx)

    # ------------------------------------------------------------------
    def _current(self, ctx: DashboardContext) -> None:
        left, right = st.columns([3, 2])

        with left:
            st.subheader("Поточний моментум, % річних")
            heat = ctx.snapshot.dropna(how="all")
            if heat.empty:
                st.info("Замало історії для обраних вікон.")
            else:
                st.plotly_chart(ctx.charts.snapshot_heatmap(heat), width="stretch")

        with right:
            window = st.selectbox("Сортувати за вікном", ctx.windows,
                                  index=len(ctx.windows) - 1,
                                  format_func=window_label, key="sort_win")
            table = ctx.snapshot.sort_values(window_label(window), ascending=False)
            st.dataframe(styled_snapshot(table), width="stretch",
                         height=60 + 35 * len(table))

    def _history(self, ctx: DashboardContext) -> None:
        st.subheader("Місце в рейтингу з часом")
        window = st.selectbox("Вікно для рейтингу", ctx.windows,
                              index=len(ctx.windows) - 1,
                              format_func=window_label, key="rank_win")
        momentum = ctx.in_range(ctx.panel[window])
        analyzer = RankingAnalyzer(momentum)

        st.plotly_chart(ctx.charts.rank_history(analyzer.ranks()), width="stretch")

        shares = analyzer.leader_shares()
        if not shares.empty:
            st.subheader(f"Частка часу в лідерах, вікно {window_label(window)}")
            st.plotly_chart(ctx.charts.leader_shares(shares), width="stretch")
