"""Cache-aware loading of candles and of aligned close-price matrices."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Sequence

import pandas as pd

from ..timeframe import Timeframe
from .cache import CandleCache
from .clients import ExchangeClient, SymbolNotFound, empty_ohlcv, get_client, utc

ProgressHook = Callable[[str, int, int], None]

RESAMPLE_RULES = {"open": "first", "high": "max", "low": "min",
                  "close": "last", "volume": "sum"}


@dataclass
class PriceMatrix:
    """Close prices for a basket of coins on one shared, regular time grid.

    `errors` maps a symbol that could not be loaded to the reason it was
    skipped — a missing pair should never take the whole basket down.
    """

    prices: pd.DataFrame
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def empty(self) -> bool:
        return self.prices.empty

    @property
    def symbols(self) -> list[str]:
        return list(self.prices.columns)

    def __len__(self) -> int:
        return len(self.prices)


class MarketDataLoader:
    """Loads OHLCV for one exchange/quote/interval combination.

    Only the ranges that are missing from the cache are actually downloaded,
    so re-opening the dashboard on a warm cache costs a single tail request
    per coin.
    """

    def __init__(
        self,
        exchange: str = "binance",
        quote: str = "USDT",
        interval: str = "4h",
        cache: CandleCache | None = None,
        use_cache: bool = True,
        client: ExchangeClient | None = None,
    ):
        # `client` is an injection point: pass one in and `exchange` is
        # ignored, which is what the tests use to avoid touching the network.
        self.client: ExchangeClient = client or get_client(exchange)
        self.exchange = self.client.name
        self.quote = quote
        self.timeframe = Timeframe(interval)
        self.cache = cache or CandleCache()
        self.use_cache = use_cache

    @property
    def interval(self) -> str:
        return self.timeframe.interval

    # ------------------------------------------------------------------
    # single pair
    # ------------------------------------------------------------------
    def ohlcv(
        self,
        base: str,
        start: str | datetime | pd.Timestamp = "2020-01-01",
        end: str | datetime | pd.Timestamp | None = None,
        refresh: bool = False,
    ) -> pd.DataFrame:
        """Full OHLCV for one pair, fetching only what the cache lacks."""
        start = utc(start)
        end = pd.Timestamp.now(tz="UTC") if end is None else utc(end)
        symbol = self.client.symbol_for(base, self.quote)
        native = self.client.native_interval(self.interval)

        stored = empty_ohlcv()
        if self.use_cache and not refresh:
            stored = self.cache.read(self.exchange, symbol, native)

        gaps = self._missing_ranges(stored, start, end, native)
        if gaps:
            stored = self._download_gaps(stored, gaps, symbol, native)
            if self.use_cache:
                self.cache.write(stored, self.exchange, symbol, native)

        if stored.empty:
            raise SymbolNotFound(f"{self.exchange}: no data for {symbol}")

        window = stored.loc[(stored.index >= start) & (stored.index <= end)]
        return window if native == self.interval else self._resample(window)

    def _missing_ranges(self, stored, start, end, native) -> list[tuple]:
        """Ranges to download: the head before the cache and the tail after it."""
        if stored.empty:
            return [(start, end)]
        step = timedelta(minutes=Timeframe(native).minutes)
        gaps = []
        if start < stored.index[0] - step:
            gaps.append((start, stored.index[0]))
        # Two steps of slack: the newest cached bar may simply be the last one
        # that has actually closed, which is not a gap worth a request.
        if end > stored.index[-1] + 2 * step:
            gaps.append((stored.index[-1], end))
        return gaps

    def _download_gaps(self, stored, gaps, symbol, native) -> pd.DataFrame:
        parts = [stored] + [self.client.fetch(symbol, native, lo, hi) for lo, hi in gaps]
        parts = [p for p in parts if not p.empty]
        if not parts:
            return empty_ohlcv()
        merged = pd.concat(parts)
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        return self._drop_unclosed(merged, native)

    @staticmethod
    def _drop_unclosed(frame: pd.DataFrame, interval: str) -> pd.DataFrame:
        """Discard the still-forming candle so a partial bar is never cached."""
        if frame.empty:
            return frame
        step = timedelta(minutes=Timeframe(interval).minutes)
        return frame[frame.index + step <= pd.Timestamp.now(tz="UTC")]

    def _resample(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        return (frame
                .resample(self.timeframe.pandas_freq, label="left", closed="left")
                .agg(RESAMPLE_RULES)
                .dropna(how="all"))

    # ------------------------------------------------------------------
    # basket
    # ------------------------------------------------------------------
    def close_matrix(
        self,
        symbols: Sequence[str],
        start="2020-01-01",
        end=None,
        refresh: bool = False,
        on_progress: ProgressHook | None = None,
    ) -> PriceMatrix:
        """Close prices for many coins, aligned on one regular grid."""
        series: dict[str, pd.Series] = {}
        errors: dict[str, str] = {}
        total = len(symbols)

        for i, symbol in enumerate(symbols, start=1):
            if on_progress:
                on_progress(symbol, i, total)
            try:
                frame = self.ohlcv(symbol, start, end, refresh=refresh)
                if frame.empty:
                    errors[symbol] = "порожня історія"
                else:
                    series[symbol] = frame["close"]
            except SymbolNotFound as exc:
                errors[symbol] = f"немає пари на біржі ({exc})"
            except Exception as exc:  # network / API hiccup: skip, keep going
                errors[symbol] = f"{type(exc).__name__}: {exc}"

        if not series:
            return PriceMatrix(pd.DataFrame(), errors)
        return PriceMatrix(self._align(series), errors)

    def _align(self, series: dict[str, pd.Series]) -> pd.DataFrame:
        prices = pd.DataFrame(series).sort_index()
        grid = pd.date_range(prices.index[0], prices.index[-1],
                             freq=self.timeframe.pandas_freq, tz="UTC")
        # A short forward fill bridges the occasional exchange outage without
        # inventing history: never more than a day of made-up prices.
        limit = max(1, int(self.timeframe.bars_per_day))
        prices = prices.reindex(grid).ffill(limit=limit)
        prices.index.name = "time"
        return prices
