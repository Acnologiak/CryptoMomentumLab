"""Warm up the parquet cache before opening the dashboard.

    python prefetch.py --interval 1h --years 5
    python prefetch.py --symbols BTC ETH SOL --interval 4h --exchange bybit

Optional: the app downloads whatever is missing on its own, this just moves
the wait out of the browser.
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

from momentumlab.config import DEFAULT_UNIVERSE
from momentumlab.data import EXCHANGES, CandleCache, CoinGeckoUniverse, MarketDataLoader
from momentumlab.timeframe import SUPPORTED_INTERVALS


class CachePrefetcher:
    """Downloads history for a list of symbols, one line of output each."""

    def __init__(self, exchange: str, quote: str, interval: str, refresh: bool = False):
        self.loader = MarketDataLoader(exchange=exchange, quote=quote, interval=interval)
        self.refresh = refresh

    def run(self, symbols: list[str], start) -> None:
        total = len(symbols)
        for i, symbol in enumerate(symbols, start=1):
            started = time.time()
            try:
                frame = self.loader.ohlcv(symbol, start, refresh=self.refresh)
                print(f"[{i}/{total}] {symbol:<6} {len(frame):>7,} bars  "
                      f"{frame.index[0]:%Y-%m-%d} -> {frame.index[-1]:%Y-%m-%d}  "
                      f"{time.time() - started:5.1f}s")
            except Exception as exc:
                print(f"[{i}/{total}] {symbol:<6} SKIPPED: {type(exc).__name__}: {exc}")


def resolve_symbols(requested: list[str] | None) -> list[str]:
    if requested:
        return [s.upper() for s in requested]
    try:
        symbols = CoinGeckoUniverse().top_symbols(10)
        print(f"top-10 by market cap: {', '.join(symbols)}")
        return symbols
    except Exception as exc:
        print(f"CoinGecko unavailable ({exc}); using static list")
        return DEFAULT_UNIVERSE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download OHLCV history into data_cache/")
    parser.add_argument("--symbols", nargs="*", default=None,
                        help="tickers; default = top-10 by market cap (CoinGecko), "
                             "falling back to a static list")
    parser.add_argument("--quote", default="USDT")
    parser.add_argument("--interval", default="1h", choices=list(SUPPORTED_INTERVALS))
    parser.add_argument("--years", type=float, default=5.0)
    parser.add_argument("--exchange", default="binance", choices=list(EXCHANGES))
    parser.add_argument("--refresh", action="store_true",
                        help="ignore the cache and refetch everything")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbols = resolve_symbols(args.symbols)
    start = (pd.Timestamp.now("UTC").normalize()
             - pd.Timedelta(days=int(args.years * 365.25)))

    CachePrefetcher(args.exchange, args.quote, args.interval, args.refresh).run(
        symbols, start)

    print()
    print(CandleCache().status().to_string(index=False))


if __name__ == "__main__":
    main()
