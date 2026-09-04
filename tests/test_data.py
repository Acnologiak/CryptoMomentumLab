"""Data-layer checks. Nothing here touches the network: the exchange client
is swapped for a stub that serves a deterministic candle generator.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from conftest import INTERVAL  # noqa: F401  (keeps sys.path patched)

from momentumlab.data import CandleCache, MarketDataLoader, StablecoinFilter
from momentumlab.data.clients import (
    BinanceClient,
    CoinbaseClient,
    ExchangeClient,
    build_ohlcv,
    utc,
)
from momentumlab.timeframe import Timeframe


class StubClient(ExchangeClient):
    """Serves synthetic hourly candles and counts how often it was called."""

    name = "stub"
    native_intervals = frozenset({"1h", "4h"})

    def __init__(self):
        self.calls: list[tuple] = []

    def fetch(self, symbol, interval, start, end):
        step = pd.Timedelta(minutes=Timeframe(interval).minutes)
        index = pd.date_range(utc(start).ceil(step), utc(end), freq=step, tz="UTC")
        self.calls.append((symbol, interval, index[0] if len(index) else None))
        rows = [(int(ts.timestamp() * 1000), 1.0, 2.0, 0.5, 1.0 + i, 10.0)
                for i, ts in enumerate(index)]
        return build_ohlcv(rows)


def loader_with_stub(tmp: Path, interval: str = "1h") -> tuple[MarketDataLoader, StubClient]:
    client = StubClient()
    loader = MarketDataLoader(quote="USDT", interval=interval,
                              cache=CandleCache(tmp), client=client)
    return loader, client


def test_pair_naming_is_venue_specific():
    assert BinanceClient().symbol_for("btc", "usdt") == "BTCUSDT"
    coinbase = CoinbaseClient()
    assert coinbase.symbol_for("btc", "USDT") == "BTC-USD"
    assert coinbase.symbol_for("eth", "USDC") == "ETH-USD"
    assert coinbase.symbol_for("eth", "BTC") == "ETH-BTC"


def test_native_interval_falls_back_to_hourly():
    assert BinanceClient().native_interval("4h") == "4h"
    coinbase = CoinbaseClient()
    assert coinbase.native_interval("6h") == "6h"
    assert coinbase.native_interval("4h") == "1h"   # resampled by the loader


def test_second_load_is_served_from_the_cache():
    with tempfile.TemporaryDirectory() as tmp:
        loader, client = loader_with_stub(Path(tmp))
        first = loader.ohlcv("BTC", start="2023-01-01", end="2023-01-10")
        calls_after_first = len(client.calls)
        second = loader.ohlcv("BTC", start="2023-01-01", end="2023-01-10")

        assert not first.empty
        assert len(client.calls) == calls_after_first, "the cache was not used"
        pd.testing.assert_frame_equal(first, second)
        assert list(Path(tmp).glob("*.parquet")), "nothing was written to disk"


def test_widening_the_range_only_fetches_the_missing_head():
    with tempfile.TemporaryDirectory() as tmp:
        loader, client = loader_with_stub(Path(tmp))
        loader.ohlcv("BTC", start="2023-01-05", end="2023-01-10")
        before = len(client.calls)
        wider = loader.ohlcv("BTC", start="2023-01-01", end="2023-01-10")

        assert len(client.calls) == before + 1, "expected exactly one gap request"
        assert wider.index[0] <= pd.Timestamp("2023-01-02", tz="UTC")


def test_unclosed_candle_is_never_cached():
    with tempfile.TemporaryDirectory() as tmp:
        loader, _ = loader_with_stub(Path(tmp))
        frame = loader.ohlcv("BTC", start=pd.Timestamp.now("UTC") - pd.Timedelta(days=2))
        step = pd.Timedelta(minutes=60)
        assert (frame.index + step <= pd.Timestamp.now("UTC")).all()


def test_non_native_interval_is_resampled_from_hourly():
    with tempfile.TemporaryDirectory() as tmp:
        loader, client = loader_with_stub(Path(tmp), interval="12h")
        frame = loader.ohlcv("BTC", start="2023-01-01", end="2023-01-05")

        assert client.calls[0][1] == "1h", "should have requested the hourly base"
        deltas = frame.index.to_series().diff().dropna().unique()
        assert list(deltas) == [pd.Timedelta(hours=12)]
        assert frame["volume"].iloc[0] == 12 * 10.0  # twelve hourly bars summed


def test_close_matrix_aligns_and_reports_failures():
    with tempfile.TemporaryDirectory() as tmp:
        loader, _ = loader_with_stub(Path(tmp))

        original_fetch = loader.client.fetch

        def fetch(symbol, interval, start, end):
            if symbol.startswith("GHOST"):
                raise RuntimeError("delisted")
            return original_fetch(symbol, interval, start, end)

        loader.client.fetch = fetch
        matrix = loader.close_matrix(["BTC", "GHOST", "ETH"],
                                     start="2023-01-01", end="2023-01-05")

        assert matrix.symbols == ["BTC", "ETH"]
        assert "GHOST" in matrix.errors and "delisted" in matrix.errors["GHOST"]
        # both survivors land on one regular, gap-free UTC grid
        steps = matrix.prices.index.to_series().diff().dropna().unique()
        assert list(steps) == [pd.Timedelta(hours=1)]
        assert matrix.prices.notna().all().all()
        assert matrix.prices.index.tz is not None


def test_cache_status_lists_written_files():
    with tempfile.TemporaryDirectory() as tmp:
        loader, _ = loader_with_stub(Path(tmp))
        loader.ohlcv("BTC", start="2023-01-01", end="2023-01-05")
        status = CandleCache(Path(tmp)).status()

        assert len(status) == 1
        assert status.iloc[0]["bars"] > 0
        assert status.iloc[0]["from"] < status.iloc[0]["to"]


def test_corrupt_cache_file_is_treated_as_cold():
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "stub_BTCUSDT_1h.parquet").write_text("not a parquet file")
        loader, client = loader_with_stub(Path(tmp))
        frame = loader.ohlcv("BTC", start="2023-01-01", end="2023-01-05")
        assert not frame.empty and client.calls


def test_stablecoin_filter():
    stables = StablecoinFilter()
    assert stables.is_stable("USDT", "Tether")
    assert stables.is_stable("WBTC", "Wrapped Bitcoin")
    assert stables.is_stable("XYZ", "Some USD Coin")
    assert not stables.is_stable("BTC", "Bitcoin")
    assert not stables.is_stable("SOL", "Solana")


def test_build_ohlcv_sorts_and_deduplicates():
    ts = int(pd.Timestamp("2023-01-01", tz="UTC").timestamp() * 1000)
    hour = 3_600_000
    rows = [(ts + hour, 1, 2, 0.5, 1.5, 5), (ts, 1, 2, 0.5, 1.0, 5),
            (ts + hour, 1, 2, 0.5, 9.9, 5)]
    frame = build_ohlcv(rows)

    assert len(frame) == 2
    assert frame.index.is_monotonic_increasing
    assert np.isclose(frame["close"].iloc[-1], 1.5)  # first duplicate wins


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            passed += 1
            print(f"ok  {name}")
    print(f"\n{passed} passed")
