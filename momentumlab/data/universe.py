"""Picking which coins to look at, via CoinGecko's public market-cap ranking."""
from __future__ import annotations

import pandas as pd
import requests

from ..config import USER_AGENT


class StablecoinFilter:
    """Drops pegged and tokenized assets — they are useless for momentum work.

    Symbols are matched exactly; names are matched on a few substrings, which
    catches the long tail of wrapped/staked/treasury products that keeps
    appearing in the top 60 without having to enumerate every ticker.
    """

    SYMBOLS = {
        "USDT", "USDC", "USDS", "DAI", "TUSD", "FDUSD", "BUSD", "USDE", "SUSDE",
        "PYUSD", "USD1", "RLUSD", "USDD", "USDP", "GUSD", "FRAX", "LUSD", "EURC",
        "USDF", "USDG", "BUIDL", "XAUT", "PAXG",
    }
    NAME_HINTS = ("usd", "dollar", "euro", "tether", "staked ether", "wrapped",
                  "heloc", "treasury", "gold")

    def is_stable(self, symbol: str, name: str = "") -> bool:
        if symbol.upper() in self.SYMBOLS:
            return True
        lowered = name.lower()
        return any(hint in lowered for hint in self.NAME_HINTS)


class CoinGeckoUniverse:
    """Top coins by market cap. No API key needed for this endpoint."""

    ENDPOINT = "https://api.coingecko.com/api/v3/coins/markets"
    SCAN_DEPTH = 60  # fetch a surplus so filtering still leaves `n` coins

    def __init__(self, stable_filter: StablecoinFilter | None = None):
        self.stable_filter = stable_filter or StablecoinFilter()

    def top_by_market_cap(self, n: int = 10, exclude_stables: bool = True) -> pd.DataFrame:
        response = requests.get(
            self.ENDPOINT,
            params={"vs_currency": "usd", "order": "market_cap_desc",
                    "per_page": self.SCAN_DEPTH, "page": 1},
            timeout=25,
            headers={"User-Agent": USER_AGENT},
        )
        response.raise_for_status()

        rows = []
        for coin in response.json():
            symbol = (coin.get("symbol") or "").upper()
            name = coin.get("name") or ""
            if exclude_stables and self.stable_filter.is_stable(symbol, name):
                continue
            if not symbol.isalnum():  # tokenized products such as FIGR_HELOC
                continue
            rows.append({"symbol": symbol, "name": name,
                         "rank": coin.get("market_cap_rank"),
                         "market_cap": coin.get("market_cap")})
            if len(rows) >= n:
                break
        return pd.DataFrame(rows)

    def top_symbols(self, n: int = 10) -> list[str]:
        return self.top_by_market_cap(n)["symbol"].tolist()
