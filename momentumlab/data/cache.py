"""On-disk parquet cache for downloaded candles.

One file per (exchange, symbol, native interval). The cache is deliberately
dumb: it stores and returns whole frames and never decides *what* is missing —
that is the loader's job.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import CACHE_DIR
from .clients import empty_ohlcv


class CandleCache:
    """Parquet-backed store of OHLCV frames."""

    def __init__(self, directory: Path | str = CACHE_DIR):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def path_for(self, exchange: str, symbol: str, interval: str) -> Path:
        safe = symbol.replace("-", "").replace("/", "")
        return self.directory / f"{exchange}_{safe}_{interval}.parquet"

    def read(self, exchange: str, symbol: str, interval: str) -> pd.DataFrame:
        return self._read_file(self.path_for(exchange, symbol, interval))

    def write(self, frame: pd.DataFrame, exchange: str, symbol: str, interval: str) -> None:
        if frame.empty:
            return
        frame.to_parquet(self.path_for(exchange, symbol, interval))

    def status(self) -> pd.DataFrame:
        """What is already on disk — used by the sidebar and the prefetch CLI."""
        rows = []
        for path in sorted(self.directory.glob("*.parquet")):
            frame = self._read_file(path)
            rows.append({
                "file": path.name,
                "bars": len(frame),
                "from": None if frame.empty else frame.index[0],
                "to": None if frame.empty else frame.index[-1],
                "MB": round(path.stat().st_size / 1e6, 2),
            })
        return pd.DataFrame(rows)

    @staticmethod
    def _read_file(path: Path) -> pd.DataFrame:
        if not path.exists():
            return empty_ohlcv()
        try:
            frame = pd.read_parquet(path)
        except Exception:
            # A half-written or version-mismatched file is not worth crashing
            # over: treat it as a cold cache and refetch.
            return empty_ohlcv()
        if frame.index.tz is None:
            frame.index = frame.index.tz_localize("UTC")
        return frame.sort_index()
