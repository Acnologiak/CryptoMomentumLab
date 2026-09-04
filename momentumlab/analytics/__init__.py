"""Momentum maths: smoothing, annualization, rankings, plot decimation."""
from .momentum import MOMENTUM_MODES, MomentumCalculator, MomentumMode, MomentumPanel
from .ranking import RankingAnalyzer
from .series import decimate
from .smoothing import (
    SMOOTHING_METHODS,
    ExponentialMovingAverage,
    PassThrough,
    SimpleMovingAverage,
    Smoother,
    Smoothing,
    ZeroLagEMA,
    get_smoother,
    smooth_prices,
)

__all__ = [
    "MOMENTUM_MODES", "SMOOTHING_METHODS", "ExponentialMovingAverage",
    "MomentumCalculator", "MomentumMode", "MomentumPanel", "PassThrough",
    "RankingAnalyzer", "SimpleMovingAverage", "Smoother", "Smoothing",
    "ZeroLagEMA", "decimate", "get_smoother", "smooth_prices",
]
