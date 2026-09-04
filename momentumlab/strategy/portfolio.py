"""Holdings, trade execution and the trade ledger.

The portfolio knows nothing about momentum: it is handed a bar position and
told what to buy, sell or rebalance to. Every leg is BTC<->alt, priced at the
bar's close with no slippage, and charged `fee_bps` of its own notional.
"""
from __future__ import annotations

import numpy as np

from .market import MarketView

BUY = "купівля"
SELL = "продаж"

#: Below this the fee model and the weight bookkeeping are noise, so a
#: rebalance leg that small is treated as "no trade".
MIN_TRADE_WEIGHT = 1e-6


class Portfolio:
    """Quantities per coin plus the journal of everything that was traded."""

    def __init__(self, market: MarketView, fee_rate: float,
                 initial_capital: float = 1.0, start_pos: int = 0):
        self.market = market
        self.fee_rate = fee_rate
        self.quantities = np.zeros(market.n_columns)
        self.quantities[market.base_pos] = (
            initial_capital / market.prices[start_pos, market.base_pos]
        )
        self.trades: list[dict] = []

    # ------------------------------------------------------------------
    # valuation
    # ------------------------------------------------------------------
    def values_at(self, pos: int) -> np.ndarray:
        """Value of every holding at `pos`, with not-yet-listed coins as 0.

        A coin that has not started trading yet has a NaN price but a zero
        quantity — it can never have been bought, since an entry requires a
        valid, non-NaN momentum. Its true contribution is therefore 0, not
        NaN. Plain `quantities * prices` would poison the whole portfolio sum
        the moment any coin in the basket is younger than the others (SOL
        listing months after BTC, say), so the product is sanitised here.
        """
        return np.nan_to_num(self.quantities * self.market.prices[pos], nan=0.0)

    def total_value(self, pos: int) -> float:
        return float(self.values_at(pos).sum())

    def base_value(self, pos: int) -> float:
        price = self.market.prices[pos, self.market.base_pos]
        return float(self.quantities[self.market.base_pos] * price)

    def weight_row(self, pos: int) -> dict:
        values = self.values_at(pos)
        total = float(values.sum())
        row = {"time": self.market.time_at(pos)}
        for coin, i in self.market.column_pos.items():
            row[coin] = values[i] / total if total > 0 else 0.0
        return row

    # ------------------------------------------------------------------
    # event-driven execution (incremental sizing)
    # ------------------------------------------------------------------
    def buy(self, pos: int, coin: str, notional: float) -> bool:
        """Spend `notional` worth of base on `coin`. Returns whether it traded."""
        prices = self.market.prices[pos]
        target = self.market.column_pos[coin]
        base = self.market.base_pos
        if notional <= 0 or not prices[target] > 0 or not prices[base] > 0:
            return False

        fee = notional * self.fee_rate
        self.quantities[base] -= notional / prices[base]
        self.quantities[target] += (notional - fee) / prices[target]
        self._record(pos, coin, BUY, notional, fee)
        return True

    def sell_all(self, pos: int, coin: str) -> bool:
        """Liquidate the whole position in `coin` back into the base asset."""
        prices = self.market.prices[pos]
        target = self.market.column_pos[coin]
        base = self.market.base_pos
        notional = self.quantities[target] * prices[target]
        if not notional > 0:
            return False

        fee = notional * self.fee_rate
        if prices[base] > 0:
            self.quantities[base] += (notional - fee) / prices[base]
        self.quantities[target] = 0.0
        self._record(pos, coin, SELL, notional, fee)
        return True

    # ------------------------------------------------------------------
    # periodic execution (target-weight sizing)
    # ------------------------------------------------------------------
    def rebalance_to(self, pos: int, targets: np.ndarray) -> None:
        """Pin every holding to its target weight of the whole portfolio.

        Fees are charged once on the round-trip turnover rather than leg by
        leg, which is the natural fit for a periodic rebalance: the cost of
        the checkpoint is proportional to how much of the book had to move.
        """
        prices = self.market.prices[pos]
        values = self.values_at(pos)
        total = float(values.sum())
        if total <= 0:
            return

        current = values / total
        deltas = targets - current
        turnover = float(np.abs(deltas[self.market.alt_pos]).sum())
        fee_total = total * turnover * self.fee_rate

        for coin, column in zip(self.market.alts, self.market.alt_pos):
            delta = deltas[column]
            if abs(delta) <= MIN_TRADE_WEIGHT:
                continue
            notional = abs(delta) * total
            self._record(pos, coin, BUY if delta > 0 else SELL,
                         notional, notional * self.fee_rate)

        net = max(total - fee_total, 0.0)
        self.quantities = np.divide(targets * net, prices,
                                    out=np.zeros(self.market.n_columns),
                                    where=prices > 0)

    # ------------------------------------------------------------------
    def _record(self, pos: int, coin: str, side: str, value: float, fee: float) -> None:
        self.trades.append({"time": self.market.time_at(pos), "coin": coin,
                            "side": side, "value": value, "fee": fee})
