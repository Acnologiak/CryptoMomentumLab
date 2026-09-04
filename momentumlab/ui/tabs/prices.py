"""Tab 3 — the underlying price series."""
from __future__ import annotations

import streamlit as st

from ..context import DashboardContext
from .base import DashboardTab


class PricesTab(DashboardTab):
    title = "💵 Ціни"

    def render(self, ctx: DashboardContext) -> None:
        normalize = st.checkbox("Нормувати до 100 на початок періоду", value=True)
        log_axis = st.checkbox("Логарифмічна вісь Y", value=True)

        prices = ctx.in_range(ctx.prices)
        if normalize and not prices.empty:
            # bfill first so a coin that starts mid-period is still rebased on
            # its own first print rather than on a NaN.
            prices = prices.div(prices.ffill().bfill().iloc[0]).mul(100)

        y_title = "=100 на старті" if normalize else f"ціна, {ctx.settings.quote}"
        st.plotly_chart(ctx.charts.price_lines(prices, log_axis, y_title),
                        width="stretch")
