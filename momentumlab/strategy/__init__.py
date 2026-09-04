"""The BTC <-> alts momentum-rotation strategy: config, engines, scoring."""
from .backtester import Backtester, run_backtest
from .config import SIZING_MODES, ExitTrigger, Sizing, StrategyConfig
from .engines import EngineOutput, IncrementalEngine, RotationEngine, TargetWeightEngine
from .market import MarketView
from .metrics import JUSTIFIED, PREMATURE, PerformanceReport, drawdown
from .portfolio import BUY, SELL, Portfolio
from .results import BacktestResult
from .signals import (
    EarlyExitRule,
    EarlyExitVerdict,
    SignalBuilder,
    SignalFrames,
    rolling_ols_slope,
)

__all__ = [
    "BUY", "JUSTIFIED", "PREMATURE", "SELL", "SIZING_MODES", "BacktestResult",
    "Backtester", "EarlyExitRule", "EarlyExitVerdict", "EngineOutput", "ExitTrigger",
    "IncrementalEngine", "MarketView", "PerformanceReport", "Portfolio",
    "RotationEngine", "SignalBuilder", "SignalFrames", "Sizing", "StrategyConfig",
    "TargetWeightEngine", "drawdown", "rolling_ols_slope", "run_backtest",
]
