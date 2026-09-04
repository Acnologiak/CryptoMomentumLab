"""Momentum-rotation backtest: BTC as the base asset, alts as rented slots.

BTC (or whatever `config.base` names) is the "stable" leg — the portfolio
sits in it by default. At every decision point each alt's annualized momentum
is compared to the base asset's; an alt earns a slot when it is far enough
ahead, and gives it back when the edge fades. Only base<->alt swaps ever
happen — an alt-to-alt replacement is executed as two legs through the base,
and pays two fees.

Momentum is computed over the *full* price history that is passed in, so the
look-back window is warmed up with real data, and only then is the simulation
restricted to [start, end].
"""
from __future__ import annotations

import pandas as pd

from ..timeframe import Timeframe
from .config import Sizing, StrategyConfig
from .engines import IncrementalEngine, RotationEngine, TargetWeightEngine
from .market import MarketView
from .results import EARLY_EXIT_COLUMNS, TRADE_COLUMNS, BacktestResult
from .signals import EarlyExitRule, SignalBuilder, SignalFrames


class Backtester:
    """Runs one `StrategyConfig` over one price history."""

    def __init__(self, config: StrategyConfig, timeframe: Timeframe | str = "4h"):
        self.config = config
        self.timeframe = timeframe if isinstance(timeframe, Timeframe) else Timeframe(timeframe)
        self.signal_builder = SignalBuilder(config, self.timeframe)

    def run(
        self,
        prices: pd.DataFrame,
        start=None,
        end=None,
        initial_capital: float = 1.0,
    ) -> BacktestResult:
        prices, alts = self._prepare(prices)
        signals = self.signal_builder.build(prices, alts)

        simulated = self._slice(prices, start, end)
        local = signals.restricted_to(simulated.index)
        market = MarketView.build(simulated, local, self.config.base, alts)

        engine = self._make_engine(market, local, initial_capital)
        output = engine.run()

        equity = pd.Series(output.equity, index=simulated.index,
                           name="стратегія") / initial_capital
        hold = simulated[self.config.base] / simulated[self.config.base].iloc[0]
        hold.name = f"утримання {self.config.base}"

        return BacktestResult(
            equity=equity,
            hold_equity=hold,
            weights=pd.DataFrame(output.weights).set_index("time"),
            active=pd.DataFrame(output.active).set_index("time"),
            trades=pd.DataFrame(output.trades, columns=TRADE_COLUMNS),
            momentum=local.momentum,
            diff=local.diff,
            early_exits=pd.DataFrame(output.early_exits, columns=EARLY_EXIT_COLUMNS),
            config=self.config,
        )

    # ------------------------------------------------------------------
    def _prepare(self, prices: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        if self.config.base not in prices.columns:
            raise ValueError(
                f"базового активу {self.config.base!r} немає серед завантажених монет"
            )
        prices = prices.dropna(how="all")
        alts = [c for c in prices.columns if c != self.config.base]
        if not alts:
            raise ValueError("потрібна хоча б одна альткоїна крім базової")
        return prices, alts

    @staticmethod
    def _slice(prices: pd.DataFrame, start, end) -> pd.DataFrame:
        if start is None and end is None:
            return prices
        low = prices.index[0] if start is None else pd.Timestamp(start)
        high = prices.index[-1] if end is None else pd.Timestamp(end)
        simulated = prices.loc[(prices.index >= low) & (prices.index <= high)]
        if simulated.empty:
            raise ValueError("порожній діапазон для симуляції")
        return simulated

    def _make_engine(self, market: MarketView, signals: SignalFrames,
                     initial_capital: float) -> RotationEngine:
        alts = market.alts
        as_array = lambda df: None if df is None else df[alts].to_numpy(dtype=float)  # noqa: E731
        early_exit = EarlyExitRule(
            self.config, self.timeframe,
            slope=as_array(signals.slope),
            fast_diff=as_array(signals.fast_diff),
            fast_momentum=as_array(signals.fast_momentum),
        )

        if self.config.sizing is Sizing.INCREMENTAL:
            return IncrementalEngine(market, self.config, early_exit, initial_capital)

        step = self.timeframe.bars(self.config.rebalance_days)
        return TargetWeightEngine(market, self.config, early_exit, initial_capital,
                                  checkpoints=market.checkpoints(step))


def run_backtest(prices, timeframe, config: StrategyConfig, start=None, end=None,
                 initial_capital: float = 1.0) -> BacktestResult:
    """One-liner for scripts and tests that do not need to keep the engine."""
    return Backtester(config, timeframe).run(prices, start, end, initial_capital)
