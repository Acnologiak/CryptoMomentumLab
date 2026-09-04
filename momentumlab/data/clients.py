"""Thin REST clients for the three public spot exchanges we support.

Each client knows only how to turn (symbol, interval, start, end) into a
DataFrame of candles indexed by UTC bar-open time. Caching, gap detection and
resampling live one level up, in `loader.py`.

Three exchanges are supported so the app keeps working from any region:
Binance is the default and the deepest, Bybit is the drop-in fallback when
Binance blocks the IP, and Coinbase works from the US but pages in tiny
300-candle chunks and only quotes USD.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from datetime import timedelta
from typing import Iterable, Sequence

import pandas as pd
import requests

from ..config import USER_AGENT
from ..timeframe import Timeframe

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


class SymbolNotFound(RuntimeError):
    """The requested pair is not listed on the selected exchange."""


def utc(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def to_millis(ts) -> int:
    return int(utc(ts).timestamp() * 1000)


def empty_ohlcv() -> pd.DataFrame:
    idx = pd.DatetimeIndex([], tz="UTC", name="time")
    return pd.DataFrame(columns=OHLCV_COLUMNS, index=idx, dtype=float)


def build_ohlcv(rows: Iterable[Sequence], ts_scale: int = 1) -> pd.DataFrame:
    """rows are (ts, open, high, low, close, volume); ts in ms unless scaled."""
    frame = pd.DataFrame(list(rows), columns=["time"] + OHLCV_COLUMNS)
    if frame.empty:
        return empty_ohlcv()
    frame["time"] = pd.to_datetime(
        frame["time"].astype("int64") * ts_scale, unit="ms", utc=True
    )
    frame[OHLCV_COLUMNS] = frame[OHLCV_COLUMNS].astype(float)
    return frame.drop_duplicates("time").set_index("time").sort_index()


class ExchangeClient(ABC):
    """Common shape of an exchange adapter."""

    name: str = ""
    #: intervals the venue serves natively; anything else is resampled from 1h
    native_intervals: frozenset[str] = frozenset()
    #: polite pause between paged requests, in seconds
    throttle: float = 0.12

    def __init__(self, session: requests.Session | None = None):
        self._session = session or self._new_session()

    @staticmethod
    def _new_session() -> requests.Session:
        session = requests.Session()
        session.headers.update({"User-Agent": USER_AGENT})
        return session

    def symbol_for(self, base: str, quote: str) -> str:
        """Venue-specific spelling of a pair, e.g. BTCUSDT or BTC-USD."""
        return f"{base.upper()}{quote.upper()}"

    def native_interval(self, interval: str) -> str:
        """The interval we actually request; the caller resamples afterwards."""
        return interval if interval in self.native_intervals else "1h"

    @abstractmethod
    def fetch(self, symbol: str, interval: str, start, end) -> pd.DataFrame:
        """Download every candle in [start, end] for one pair."""


class BinanceClient(ExchangeClient):
    name = "binance"
    native_intervals = frozenset({"1h", "2h", "4h", "6h", "12h", "1d"})
    ENDPOINT = "https://api.binance.com/api/v3/klines"
    PAGE = 1000

    def fetch(self, symbol: str, interval: str, start, end) -> pd.DataFrame:
        step = Timeframe(interval).milliseconds
        cursor, end_ms, rows = to_millis(start), to_millis(end), []
        while cursor < end_ms:
            response = self._session.get(
                self.ENDPOINT,
                params={"symbol": symbol, "interval": interval, "startTime": cursor,
                        "endTime": end_ms, "limit": self.PAGE},
                timeout=25,
            )
            if response.status_code == 400:
                raise SymbolNotFound(f"binance: {symbol} not listed")
            response.raise_for_status()
            page = response.json()
            if not page:
                break
            rows.extend((c[0], c[1], c[2], c[3], c[4], c[5]) for c in page)
            cursor = int(page[-1][0]) + step
            if len(page) < self.PAGE:
                break
            time.sleep(self.throttle)
        return build_ohlcv(rows)


class BybitClient(ExchangeClient):
    name = "bybit"
    native_intervals = frozenset({"1h", "2h", "4h", "6h", "12h", "1d"})
    ENDPOINT = "https://api.bybit.com/v5/market/kline"
    PAGE = 1000
    # Bybit spells intervals in minutes, with a letter for the daily and up.
    INTERVAL_CODES = {"1h": "60", "2h": "120", "4h": "240",
                      "6h": "360", "12h": "720", "1d": "D"}

    def fetch(self, symbol: str, interval: str, start, end) -> pd.DataFrame:
        step = Timeframe(interval).milliseconds
        cursor, end_ms, rows = to_millis(start), to_millis(end), []
        while cursor < end_ms:
            response = self._session.get(
                self.ENDPOINT,
                params={"category": "spot", "symbol": symbol,
                        "interval": self.INTERVAL_CODES[interval],
                        "start": cursor, "end": end_ms, "limit": self.PAGE},
                timeout=25,
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get("retCode") not in (0, None):
                message = str(payload.get("retMsg", ""))
                if "symbol" in message.lower() or "not supported" in message.lower():
                    raise SymbolNotFound(f"bybit: {symbol} -> {message}")
                raise RuntimeError(f"bybit error: {message}")
            page = payload["result"]["list"]  # newest first
            if not page:
                break
            rows.extend((c[0], c[1], c[2], c[3], c[4], c[5]) for c in page)
            cursor = max(int(c[0]) for c in page) + step
            if len(page) < self.PAGE:
                break
            time.sleep(self.throttle)
        return build_ohlcv(rows)


class CoinbaseClient(ExchangeClient):
    name = "coinbase"
    native_intervals = frozenset({"1h", "6h", "1d"})
    ENDPOINT = "https://api.exchange.coinbase.com/products/{symbol}/candles"
    PAGE = 300
    throttle = 0.25
    GRANULARITY = {"1h": 3600, "6h": 21600, "1d": 86400}
    USD_LIKE = {"USDT", "USD", "USDC"}

    def symbol_for(self, base: str, quote: str) -> str:
        # Coinbase quotes in USD rather than USDT, and hyphenates the pair.
        quote = quote.upper()
        return f"{base.upper()}-{'USD' if quote in self.USD_LIKE else quote}"

    def fetch(self, symbol: str, interval: str, start, end) -> pd.DataFrame:
        granularity = self.GRANULARITY[interval]
        span = timedelta(seconds=granularity * self.PAGE)
        cursor, end_ts, rows = utc(start), utc(end), []
        while cursor < end_ts:
            stop = min(cursor + span, end_ts)
            response = self._session.get(
                self.ENDPOINT.format(symbol=symbol),
                params={"granularity": granularity,
                        "start": cursor.isoformat(), "end": stop.isoformat()},
                timeout=25,
            )
            if response.status_code == 404:
                raise SymbolNotFound(f"coinbase: {symbol} not listed")
            response.raise_for_status()
            # Coinbase returns [time, low, high, open, close, volume], newest first.
            rows.extend((c[0], c[3], c[2], c[1], c[4], c[5]) for c in response.json())
            cursor = stop
            time.sleep(self.throttle)
        return build_ohlcv(rows, ts_scale=1000)


_CLIENT_TYPES: dict[str, type[ExchangeClient]] = {
    cls.name: cls for cls in (BinanceClient, BybitClient, CoinbaseClient)
}

EXCHANGES = tuple(_CLIENT_TYPES)


def get_client(exchange: str, session: requests.Session | None = None) -> ExchangeClient:
    try:
        return _CLIENT_TYPES[exchange](session)
    except KeyError:
        raise ValueError(
            f"unknown exchange {exchange!r}; expected one of {', '.join(EXCHANGES)}"
        ) from None
