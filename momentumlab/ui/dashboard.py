"""Wiring: sidebar -> data -> momentum -> tabs."""
from __future__ import annotations

from datetime import timedelta

import pandas as pd
import streamlit as st

from ..timeframe import window_label
from .charts import ChartFactory
from .context import DashboardContext
from .services import load_prices
from .settings import AppSettings
from .sidebar import Sidebar
from .tabs import ALL_TABS
from .theme import color_map

PAGE_TITLE = "Crypto Momentum Lab"


class Dashboard:
    """The Streamlit app itself. `Dashboard().run()` is the whole entry point."""

    def __init__(self):
        self.sidebar = Sidebar()
        self.tabs = [tab_type() for tab_type in ALL_TABS]

    def run(self) -> None:
        self._configure_page()
        settings = self.sidebar.render()
        if not self._has_selection(settings):
            return

        matrix = load_prices(tuple(settings.coins), settings.quote, settings.interval,
                             settings.start_date, settings.exchange, settings.cache_bust)
        if matrix.empty:
            st.error("Не вдалося завантажити жодного ряду.")
            st.json(matrix.errors)
            return
        if matrix.errors:
            st.warning("Пропущено: " + "; ".join(
                f"**{k}** — {v}" for k, v in matrix.errors.items()))

        ctx = self._build_context(settings, matrix)
        self._header(ctx)
        self._render_tabs(ctx)

    # ------------------------------------------------------------------
    @staticmethod
    def _configure_page() -> None:
        st.set_page_config(page_title=PAGE_TITLE, page_icon="📈", layout="wide")

    @staticmethod
    def _has_selection(settings: AppSettings) -> bool:
        if not settings.coins:
            st.info("Оберіть хоча б одну монету у бічній панелі.")
            return False
        if not settings.windows:
            st.info("Оберіть хоча б одне вікно моментуму.")
            return False
        return True

    def _build_context(self, settings: AppSettings, matrix) -> DashboardContext:
        prices = matrix.prices
        panel = settings.calculator().panel(prices, settings.windows)
        low, high = self._period_picker(prices, settings)

        snapshot = panel.before(high).snapshot()
        snapshot = snapshot.loc[[c for c in settings.coins if c in snapshot.index]]

        colors = color_map(prices.columns)
        return DashboardContext(
            settings=settings, prices=prices, errors=matrix.errors, panel=panel,
            snapshot=snapshot, colors=colors,
            charts=ChartFactory(colors, settings.max_points), low=low, high=high,
        )

    @staticmethod
    def _period_picker(prices: pd.DataFrame, settings: AppSettings):
        """Date-range slider over the loaded history; returns [low, high)."""
        first, last = prices.index[0].date(), prices.index[-1].date()
        default_first = max(first, last - timedelta(days=settings.history_days))
        date_from, date_to = st.slider(
            "Період відображення", min_value=first, max_value=last,
            value=(default_first, last), format="YYYY-MM-DD")
        return (pd.Timestamp(date_from, tz="UTC"),
                pd.Timestamp(date_to, tz="UTC") + pd.Timedelta(days=1))

    @staticmethod
    def _header(ctx: DashboardContext) -> None:
        cells = st.columns(4)
        cells[0].metric("Монет у вибірці", len(ctx.prices.columns))
        cells[1].metric("Свічок", f"{len(ctx.prices):,}".replace(",", " "))
        cells[2].metric("Історія з", str(ctx.prices.index[0].date()))

        longest = window_label(ctx.windows[-1])
        if longest in ctx.snapshot and ctx.snapshot[longest].notna().any():
            leader = ctx.snapshot[longest].idxmax()
            cells[3].metric(f"Лідер за {longest}", leader,
                            f"{ctx.snapshot.loc[leader, longest]:+.0f}% р.р.")

    def _render_tabs(self, ctx: DashboardContext) -> None:
        containers = st.tabs([tab.title for tab in self.tabs])
        for container, tab in zip(containers, self.tabs):
            with container:
                tab.render(ctx)
