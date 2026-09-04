"""Colours and the plotly defaults every chart in the app starts from."""
from __future__ import annotations

from typing import Iterable

PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#EECA3B",
           "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC", "#1F77B4", "#8C564B"]

PLOT_TEMPLATE = "plotly_white"

STRATEGY_COLOR = "#4C78A8"
HOLD_COLOR = "#9D755D"
GAIN_COLOR = "#54A24B"
LOSS_COLOR = "#E45756"
GRID_COLOR = "#888"


def color_map(symbols: Iterable[str]) -> dict[str, str]:
    """Stable coin -> colour assignment, so a coin keeps its line colour."""
    return {symbol: PALETTE[i % len(PALETTE)] for i, symbol in enumerate(symbols)}


def layout(height: int, **overrides) -> dict:
    """Common layout arguments; anything passed in wins."""
    base = {
        "height": height,
        "template": PLOT_TEMPLATE,
        "hovermode": "x unified",
        "margin": dict(l=60, r=20, t=20, b=40),
    }
    base.update(overrides)
    return base
