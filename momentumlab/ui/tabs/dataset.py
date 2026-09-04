"""Tab 5 — raw numbers, CSV export and the state of the disk cache."""
from __future__ import annotations

import streamlit as st

from ...timeframe import window_label
from ..context import DashboardContext
from ..services import cache_directory, cache_status
from .base import DashboardTab


class DataTab(DashboardTab):
    title = "🧾 Дані"

    def render(self, ctx: DashboardContext) -> None:
        settings = ctx.settings

        st.subheader("Знімок моментуму")
        st.dataframe(ctx.snapshot.round(1), width="stretch")

        window = st.selectbox("Вікно для експорту", ctx.windows,
                              index=len(ctx.windows) - 1,
                              format_func=window_label, key="export_win")
        label = window_label(window)
        st.download_button(
            f"⬇ CSV: моментум {label}",
            ctx.in_range(ctx.panel[window]).to_csv().encode(),
            file_name=f"momentum_{settings.exchange}_{settings.interval}_{label}.csv",
            mime="text/csv",
        )
        st.download_button(
            "⬇ CSV: ціни",
            ctx.in_range(ctx.prices).to_csv().encode(),
            file_name=f"prices_{settings.exchange}_{settings.interval}.csv",
            mime="text/csv",
        )

        with st.expander("Кеш на диску"):
            st.dataframe(cache_status(), width="stretch")
            st.caption(f"Каталог: `{cache_directory()}`")
