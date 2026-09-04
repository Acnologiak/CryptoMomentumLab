"""The sidebar: everything that decides *what* is loaded and measured."""
from __future__ import annotations

import streamlit as st

from ..analytics import MOMENTUM_MODES, SMOOTHING_METHODS, MomentumMode, Smoothing
from ..config import (
    DEFAULT_EXCHANGE,
    DEFAULT_HISTORY_YEARS,
    DEFAULT_INTERVAL,
    DEFAULT_MAX_PLOT_POINTS,
    DEFAULT_MOMENTUM_MODE,
    DEFAULT_QUOTE,
    DEFAULT_SMOOTHING,
    DEFAULT_SMOOTHING_DAYS,
    DEFAULT_UNIVERSE,
    DEFAULT_WINDOWS,
    QUOTE_CURRENCIES,
    WINDOW_CHOICES,
)
from ..data import EXCHANGES
from ..timeframe import SUPPORTED_INTERVALS, window_label
from .services import load_top_symbols
from .settings import AppSettings

#: widget keys holding a window value; dropped when that window disappears
WINDOW_BOUND_KEYS = ("sort_win", "rank_win", "export_win")


class Sidebar:
    """Renders the control panel and hands back an `AppSettings`."""

    def render(self) -> AppSettings:
        self._init_state()
        settings = AppSettings(cache_bust=st.session_state.bust)

        with st.sidebar:
            self._data_section(settings)
            self._universe_section(settings)
            self._momentum_section(settings)
            self._display_section(settings)

        self._forget_stale_windows(settings.windows)
        return settings

    # ------------------------------------------------------------------
    @staticmethod
    def _init_state() -> None:
        st.session_state.setdefault("universe", DEFAULT_UNIVERSE.copy())
        st.session_state.setdefault("bust", 0)

    def _data_section(self, settings: AppSettings) -> None:
        st.header("Дані")
        settings.exchange = st.selectbox(
            "Біржа", EXCHANGES, index=EXCHANGES.index(DEFAULT_EXCHANGE),
            help="Binance блокує US-IP — тоді Bybit або Coinbase.")
        settings.quote = st.selectbox("Котирування", QUOTE_CURRENCIES,
                                      index=QUOTE_CURRENCIES.index(DEFAULT_QUOTE))
        settings.interval = st.selectbox(
            "Інтервал свічок", SUPPORTED_INTERVALS,
            index=SUPPORTED_INTERVALS.index(DEFAULT_INTERVAL))
        settings.years = st.slider("Глибина історії, років", 0.5, 8.0,
                                   DEFAULT_HISTORY_YEARS, 0.5)
        if st.button("↻ Дотягнути свіжі свічки", width="stretch"):
            st.session_state.bust += 1
            settings.cache_bust = st.session_state.bust
            st.cache_data.clear()

    def _universe_section(self, settings: AppSettings) -> None:
        st.header("Монети")
        left, right = st.columns(2)
        if left.button("Топ-10 CoinGecko", width="stretch"):
            try:
                st.session_state.universe = load_top_symbols(10)
            except Exception as exc:
                st.warning(f"CoinGecko недоступний: {exc}")
        if right.button("Скинути", width="stretch"):
            st.session_state.universe = DEFAULT_UNIVERSE.copy()

        extra = st.text_input("Додати тікери (через кому)", placeholder="AVAX, DOT, XMR")
        for symbol in self._split_tickers(extra):
            if symbol not in st.session_state.universe:
                st.session_state.universe.append(symbol)

        settings.coins = st.multiselect("Показати", st.session_state.universe,
                                        default=st.session_state.universe)

    def _momentum_section(self, settings: AppSettings) -> None:
        st.header("Моментум")
        settings.smoothing = Smoothing(st.radio(
            "Згладжування ціни", SMOOTHING_METHODS, horizontal=True,
            index=SMOOTHING_METHODS.index(DEFAULT_SMOOTHING),
            help="ZLEMA — EMA з компенсацією власної затримки: на локально лінійному "
                 "тренді майже прибирає лаг, на різких розворотах трохи перелітає. "
                 "Застосовується і до кривих, і до симуляції."))
        settings.smoothing_days = st.number_input(
            "Період згладжування, днів", 0.5, 120.0, DEFAULT_SMOOTHING_DAYS, 0.5,
            disabled=settings.smoothing is Smoothing.NONE)

        windows = st.multiselect(
            "Вікна моментуму (днів)", WINDOW_CHOICES, default=DEFAULT_WINDOWS,
            format_func=lambda d: f"{window_label(d)} ({d:g}д)")
        custom = st.text_input("Своє вікно, днів", placeholder="напр. 45, 120")
        for token in [t.strip() for t in custom.split(",") if t.strip()]:
            try:
                windows.append(float(token))
            except ValueError:
                st.warning(f"Не число: {token}")
        settings.windows = sorted(set(windows))

        settings.mode = MomentumMode(st.radio(
            "Річна нормалізація", MOMENTUM_MODES, horizontal=True,
            index=MOMENTUM_MODES.index(DEFAULT_MOMENTUM_MODE),
            format_func=lambda m: "складна (CAGR)" if m == "compound" else "логарифмічна"))

    def _display_section(self, settings: AppSettings) -> None:
        st.header("Відображення")
        settings.clip_axis = st.checkbox("Обрізати вісь Y", value=True)
        settings.clip_limit = st.slider("Межа осі Y, % річних", 100, 5000, 1000, 100,
                                        disabled=not settings.clip_axis)
        settings.max_points = st.select_slider(
            "Точок на криву", [500, 1000, 2000, 5000, 10000], DEFAULT_MAX_PLOT_POINTS)
        settings.show_zero = st.checkbox("Лінія 0%", value=True)

    # ------------------------------------------------------------------
    @staticmethod
    def _split_tickers(raw: str) -> list[str]:
        return [token.strip().upper() for token in raw.split(",") if token.strip()]

    @staticmethod
    def _forget_stale_windows(windows: list[float]) -> None:
        """Drop widget state pointing at a window the user just removed."""
        for key in WINDOW_BOUND_KEYS:
            if st.session_state.get(key) not in windows:
                st.session_state.pop(key, None)
