"""The two rotation engines.

Both share the same signal: an alt earns a slot when its annualized momentum
beats the base asset's by more than `entry_edge` (and is positive in its own
right), and keeps the slot until that edge drops below `exit_edge` or turns
negative. They differ only in how they size and maintain positions — see
`Sizing` in `config.py`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from .config import StrategyConfig
from .market import MarketView
from .portfolio import Portfolio
from .signals import EarlyExitRule


@dataclass
class EngineOutput:
    equity: np.ndarray
    weights: list[dict] = field(default_factory=list)
    active: list[dict] = field(default_factory=list)
    trades: list[dict] = field(default_factory=list)
    early_exits: list[dict] = field(default_factory=list)


class RotationEngine(ABC):
    """Shared plumbing: signal masks, early exits, bookkeeping rows."""

    def __init__(
        self,
        market: MarketView,
        config: StrategyConfig,
        early_exit: EarlyExitRule,
        initial_capital: float = 1.0,
        start_pos: int = 0,
    ):
        self.market = market
        self.config = config
        self.early_exit = early_exit
        self.portfolio = Portfolio(market, config.fee_rate, initial_capital, start_pos)
        self.state = np.zeros(market.n_alts, dtype=bool)
        self.equity = np.empty(market.n_bars)
        self.weight_rows: list[dict] = []
        self.active_rows: list[dict] = []
        self.early_exit_rows: list[dict] = []

    @abstractmethod
    def run(self) -> EngineOutput:
        """Walk the price history and return everything the report needs."""

    # ------------------------------------------------------------------
    # signals
    # ------------------------------------------------------------------
    def _entry_and_stay(self, pos: int) -> tuple[np.ndarray, np.ndarray]:
        """Which alts qualify to be opened, and which qualify to be kept."""
        diff_row = self.market.diff[pos]
        momentum_row = self.market.alt_momentum[pos]
        valid = ~np.isnan(diff_row) & ~np.isnan(momentum_row)
        positive = valid & (momentum_row > 0)
        entered = positive & (diff_row > self.config.entry_edge)
        stayed = positive & (diff_row > self.config.exit_edge)
        return entered, stayed

    def _apply_early_exit(self, pos: int, entered: np.ndarray, stayed: np.ndarray):
        """Force the accelerators' exits and withhold their re-entries.

        A coin the rule would unwind on the very next bar is not worth
        re-opening now, so while the condition holds the alt is kept out
        entirely. Only exits of positions that were actually held get logged —
        withheld entries are not exits.
        """
        if not self.early_exit.enabled:
            return entered, stayed

        diff_row = self.market.diff[pos]
        verdict = self.early_exit.evaluate(pos, diff_row)
        if not verdict.any():
            return entered, stayed

        for i in np.where(verdict.fired & self.state)[0]:
            self.early_exit_rows.append({
                "time": self.market.time_at(pos),
                "coin": self.market.alts[i],
                "trigger": str(verdict.triggers[i]),
                "diff": float(diff_row[i]),
                "predicted_days": float(verdict.predicted_days[i]),
            })
        return entered & ~verdict.fired, stayed & ~verdict.fired

    def _strongest(self, indices: np.ndarray, diff_row: np.ndarray) -> np.ndarray:
        """`indices` reordered by edge over the base asset, strongest first."""
        return indices[np.argsort(-diff_row[indices])]

    # ------------------------------------------------------------------
    # bookkeeping
    # ------------------------------------------------------------------
    def _record_bar(self, pos: int) -> None:
        self.weight_rows.append(self.portfolio.weight_row(pos))
        self.active_rows.append({
            "time": self.market.time_at(pos),
            **{coin: bool(self.state[i]) for i, coin in enumerate(self.market.alts)},
        })

    def _output(self) -> EngineOutput:
        return EngineOutput(equity=self.equity, weights=self.weight_rows,
                            active=self.active_rows, trades=self.portfolio.trades,
                            early_exits=self.early_exit_rows)


class TargetWeightEngine(RotationEngine):
    """Periodic rebalance to a fixed 1/K share of the whole portfolio.

    Every checkpoint pins each active alt back to its target weight, even ones
    that were already open — which is exactly why this mode trades (and pays)
    far more than the event-driven one.
    """

    def __init__(self, market, config, early_exit, initial_capital=1.0,
                 checkpoints: list[int] | None = None):
        self.checkpoints = checkpoints or [0]
        super().__init__(market, config, early_exit, initial_capital,
                         start_pos=self.checkpoints[0])

    def run(self) -> EngineOutput:
        slot_weight = 1.0 / self.config.max_active

        for step, pos in enumerate(self.checkpoints):
            if step > 0 and self.portfolio.total_value(pos) > 0:
                self._advance_state(pos)
                self.portfolio.rebalance_to(pos, self._targets(slot_weight))

            self._record_bar(pos)
            self._fill_equity(step, pos)

        return self._output()

    def _advance_state(self, pos: int) -> None:
        entered, stayed = self._entry_and_stay(pos)
        entered, stayed = self._apply_early_exit(pos, entered, stayed)
        state = np.where(self.state, stayed, entered)

        # More qualified coins than slots: keep the ones furthest ahead of BTC.
        overflow = state.sum() - self.config.max_active
        if overflow > 0:
            ranked = self._strongest(np.where(state)[0], self.market.diff[pos])
            state[ranked[self.config.max_active:]] = False
        self.state = state

    def _targets(self, slot_weight: float) -> np.ndarray:
        targets = np.zeros(self.market.n_columns)
        for i in np.where(self.state)[0]:
            targets[self.market.alt_pos[i]] = slot_weight
        targets[self.market.base_pos] = 1.0 - slot_weight * self.state.sum()
        return targets

    def _fill_equity(self, step: int, pos: int) -> None:
        """Mark the book to market on every bar until the next checkpoint."""
        following = self.checkpoints[step + 1] if step + 1 < len(self.checkpoints) else self.market.n_bars
        prices = np.nan_to_num(self.market.prices[pos:following], nan=0.0)
        self.equity[pos:following] = prices @ self.portfolio.quantities


class IncrementalEngine(RotationEngine):
    """Event-driven: checked every bar, trades only when the signal changes.

    Existing positions are never resized by later signals. An entry spends
    `base_balance / free_slots` — with K=3 the first entry out of all-BTC
    takes 33%, the next (2 slots free) takes 50% of what is left, and the last
    takes the remainder. An exit always sells 100% of that one position.
    """

    def run(self) -> EngineOutput:
        for pos in range(self.market.n_bars):
            if pos > 0:
                self._trade_bar(pos)
            self._record_bar(pos)
            self.equity[pos] = self.portfolio.total_value(pos)
        return self._output()

    def _trade_bar(self, pos: int) -> None:
        entered, stayed = self._entry_and_stay(pos)
        entered, stayed = self._apply_early_exit(pos, entered, stayed)

        qualified = np.where(self.state, stayed, entered)
        kept = self.state & qualified
        exiting = self.state & ~qualified
        candidates = ~self.state & qualified

        if self.config.rotation_margin is not None:
            kept, exiting = self._rotate(pos, kept, exiting, candidates)

        for i in np.where(exiting)[0]:
            self.portfolio.sell_all(pos, self.market.alts[i])

        self.state = self._fill_slots(pos, kept, candidates)

    def _rotate(self, pos, kept, exiting, candidates):
        """Evict the weakest holders in favour of much stronger outsiders.

        Without this a held coin keeps its slot until it fails the stay bar on
        its own, no matter how far ahead a coin sitting outside the K slots
        is. The margin is what keeps two near-identical coins from swapping
        rank — and trading — every bar on pure noise.
        """
        diff_row = self.market.diff[pos]
        held = np.where(kept)[0]
        contenders = np.where(candidates)[0]
        free_slots = self.config.max_active - held.size
        if contenders.size <= max(free_slots, 0):
            return kept, exiting  # everyone fits; nothing to contest

        # Weakest held first vs. strongest outsider first. Both lists are
        # sorted, so the margin check is monotonic (the kept edge only grows,
        # the candidate edge only shrinks) — the first failed pair means every
        # later pair fails too, which makes stopping there exact, not a guess.
        weakest_first = held[np.argsort(diff_row[held])]
        strongest_first = contenders[np.argsort(-diff_row[contenders])]
        contesting = strongest_first[max(free_slots, 0):]

        evictions = 0
        for j in range(min(weakest_first.size, contesting.size)):
            challenger, incumbent = contesting[j], weakest_first[j]
            if diff_row[challenger] > diff_row[incumbent] + self.config.rotation_margin:
                evictions = j + 1
            else:
                break

        for j in range(evictions):
            loser = weakest_first[j]
            kept[loser] = False
            exiting[loser] = True
        return kept, exiting

    def _fill_slots(self, pos, kept, candidates) -> np.ndarray:
        state = kept.copy()
        active = int(kept.sum())
        contenders = np.where(candidates)[0]
        if not contenders.size:
            return state

        for i in self._strongest(contenders, self.market.diff[pos]):
            free_slots = self.config.max_active - active
            if free_slots <= 0:
                break
            notional = self.portfolio.base_value(pos) / free_slots
            if self.portfolio.buy(pos, self.market.alts[i], notional):
                state[i] = True
                active += 1
        return state
