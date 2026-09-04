"""Streamlit-cached wrappers around the data layer.

`st.cache_data` keys on the arguments, so everything here takes plain
hashable values rather than loader objects. `cache_bust` is a counter the
sidebar's refresh button increments — bumping it is what makes an otherwise
identical call miss the cache.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from ..data import CandleCache, CoinGeckoUniverse, MarketDataLoader, PriceMatrix


@st.cache_data(show_spinner=False, ttl=3600)
def load_prices(
    symbols: tuple[str, ...],
    quote: str,
    interval: str,
    start: str,
    exchange: str,
    cache_bust: int = 0,
) -> PriceMatrix:
    """Close-price matrix for a basket, with a live progress bar."""
    placeholder = st.empty()
    bar = placeholder.progress(0.0, text="Завантаження…")

    def report(symbol: str, done: int, total: int) -> None:
        bar.progress(done / total, text=f"Завантаження {symbol} ({done}/{total})")

    loader = MarketDataLoader(exchange=exchange, quote=quote, interval=interval)
    try:
        return loader.close_matrix(symbols, start=start, on_progress=report)
    finally:
        placeholder.empty()


@st.cache_data(show_spinner=False, ttl=1800)
def load_top_symbols(n: int) -> list[str]:
    return CoinGeckoUniverse().top_symbols(n)


def cache_status() -> pd.DataFrame:
    """Not cached on purpose — it is a directory listing, and it changes."""
    return CandleCache().status()


def cache_directory() -> str:
    return str(CandleCache().directory)
