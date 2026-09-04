"""Small helpers for reshaping series before they hit a chart."""
from __future__ import annotations

import numpy as np
import pandas as pd


def decimate(frame: pd.DataFrame | pd.Series, max_points: int = 2000):
    """Thin a long series for plotting, always keeping the newest observation.

    Five years of hourly candles is ~44k points per coin; the browser does not
    thank you for that, and at screen resolution nothing is lost by taking
    every n-th sample. The last row is appended back so the chart always ends
    on the real, current value rather than wherever the stride happened to land.
    """
    if len(frame) <= max_points:
        return frame
    step = int(np.ceil(len(frame) / max_points))
    thinned = frame.iloc[::step]
    if thinned.index[-1] != frame.index[-1]:
        thinned = pd.concat([thinned, frame.iloc[[-1]]])
    return thinned
