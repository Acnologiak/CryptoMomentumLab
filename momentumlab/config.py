"""Project-wide paths and the defaults the dashboard starts up with."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "data_cache"

USER_AGENT = "crypto-momentum-lab/1.0"

# Quote currencies offered in the sidebar. Coinbase rewrites the USD-ish ones
# to plain USD itself, so the list stays the same for every exchange.
QUOTE_CURRENCIES = ("USDT", "USDC", "USD", "BTC")

DEFAULT_UNIVERSE = ["BTC", "ETH", "BNB", "XRP", "SOL", "TRX", "DOGE", "ADA", "LINK", "AVAX"]

# --- what the sidebar is pre-set to on a cold start ---
DEFAULT_EXCHANGE = "binance"
DEFAULT_QUOTE = "USDT"
DEFAULT_INTERVAL = "2h"
DEFAULT_HISTORY_YEARS = 5.0
DEFAULT_SMOOTHING = "SMA"
DEFAULT_SMOOTHING_DAYS = 7.0
DEFAULT_MOMENTUM_MODE = "log"
DEFAULT_MAX_PLOT_POINTS = 2000

# Look-back windows offered as checkboxes; the user can type extra ones.
WINDOW_CHOICES = [1.0, 2.0, 3.0, 7.0, 14.0, 20.0, 30.0, 60.0, 90.0, 180.0, 365.0, 730.0]
DEFAULT_WINDOWS = [7.0, 30.0, 90.0, 365.0]

# --- simulation tab ---
# A wide grid search over BTC/ETH/SOL/BNB put the sturdiest (train+test)
# configuration at EMA-7d, 20d window, entry 15pp, exit 0pp, K=2, no rotation.
# The pool and the window below are the working preset, deliberately a little
# wider than that: five liquid alts on a 14d window. Entry/exit/K are the
# grid-search values.
DEFAULT_SIM_POOL = ["ETH", "SOL", "BNB", "XRP", "TRX"]
DEFAULT_SIM_WINDOW = 14.0
DEFAULT_SIM_ENTRY_EDGE = 15.0
DEFAULT_SIM_SLOTS = 2
