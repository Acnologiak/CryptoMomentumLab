"""Tab 4 — the BTC <-> alts rotation backtest."""
from __future__ import annotations

import streamlit as st

from ...strategy import Backtester
from ..context import DashboardContext
from ..strategy_form import StrategyForm
from ..strategy_report import StrategyReportView
from .base import DashboardTab

BASE_ASSET = "BTC"


class SimulationTab(DashboardTab):
    title = "🤖 Симуляція стратегії"

    def render(self, ctx: DashboardContext) -> None:
        st.subheader("Ротація BTC ↔ альткоїни за моментумом")

        if BASE_ASSET not in ctx.prices.columns:
            st.warning(
                f"Потрібен {BASE_ASSET} у списку монет («Показати» в бічній панелі) — "
                "він базовий (стабільний) актив стратегії.")
            return

        alt_universe = [c for c in ctx.coins if c != BASE_ASSET]
        if not alt_universe:
            st.info("Додайте в бічній панелі хоча б одну альткоїну крім BTC.")
            return

        choices = StrategyForm(BASE_ASSET, alt_universe).render()
        if not choices.pool:
            st.info("Пул альткоїнів порожній — нічого симулювати.")
            return

        settings = ctx.settings
        try:
            config = choices.to_config(settings.smoothing, settings.smoothing_days,
                                       settings.mode)
            result = Backtester(config, settings.timeframe).run(
                ctx.prices[[BASE_ASSET] + choices.pool], start=ctx.low, end=ctx.high)
        except ValueError as exc:
            st.error(str(exc))
            return

        StrategyReportView(result, ctx.charts).render()
