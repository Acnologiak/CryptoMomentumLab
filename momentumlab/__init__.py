"""Crypto momentum research toolkit.

Three layers, each usable on its own:

* `momentumlab.data`      — exchange clients, parquet cache, price matrices
* `momentumlab.analytics` — smoothing, annualized momentum, rankings
* `momentumlab.strategy`  — the BTC <-> alts rotation backtest

`momentumlab.ui` sits on top of all three and is only imported by the
Streamlit entry point.
"""
from .analytics import (
    MOMENTUM_MODES,
    SMOOTHING_METHODS,
    MomentumCalculator,
    MomentumMode,
    MomentumPanel,
    RankingAnalyzer,
    Smoothing,
    decimate,
    smooth_prices,
)
from .data import (
    EXCHANGES,
    CandleCache,
    CoinGeckoUniverse,
    MarketDataLoader,
    PriceMatrix,
    SymbolNotFound,
)
from .strategy import (
    BacktestResult,
    Backtester,
    PerformanceReport,
    Sizing,
    StrategyConfig,
    drawdown,
    run_backtest,
)
from .timeframe import SUPPORTED_INTERVALS, Timeframe, window_label

__version__ = "1.0.0"

__all__ = [
    "EXCHANGES", "MOMENTUM_MODES", "SMOOTHING_METHODS", "SUPPORTED_INTERVALS",
    "BacktestResult", "Backtester", "CandleCache", "CoinGeckoUniverse",
    "MarketDataLoader", "MomentumCalculator", "MomentumMode", "MomentumPanel",
    "PerformanceReport", "PriceMatrix", "RankingAnalyzer", "Sizing", "Smoothing",
    "StrategyConfig", "SymbolNotFound", "Timeframe", "decimate", "drawdown",
    "run_backtest", "smooth_prices", "window_label",
]
