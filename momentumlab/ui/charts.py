"""Every plotly figure the dashboard draws, in one place.

The factory holds the two things every chart needs — the coin colour map and
how many points a curve may carry — so the tabs stay free of styling detail.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..analytics import decimate
from ..timeframe import window_label
from .theme import (
    GAIN_COLOR,
    GRID_COLOR,
    HOLD_COLOR,
    LOSS_COLOR,
    PLOT_TEMPLATE,
    STRATEGY_COLOR,
    layout,
)


class ChartFactory:
    def __init__(self, colors: dict[str, str], max_points: int = 2000):
        self.colors = colors
        self.max_points = max_points

    def _color(self, coin: str) -> str:
        return self.colors.get(coin, STRATEGY_COLOR)

    def _thin(self, frame):
        return decimate(frame, self.max_points)

    # ------------------------------------------------------------------
    # momentum curves
    # ------------------------------------------------------------------
    def momentum_curves(self, panel_by_window: dict[float, pd.DataFrame],
                        show_zero: bool, clip: int | None) -> go.Figure:
        """One stacked subplot per window, sharing the x-axis and the legend."""
        windows = sorted(panel_by_window)
        rows = len(windows)
        fig = make_subplots(
            rows=rows, cols=1, shared_xaxes=True,
            vertical_spacing=0.06 / max(1, rows / 2),
            subplot_titles=[f"Вікно {window_label(w)} ({w:g} днів)" for w in windows],
        )
        for row, window in enumerate(windows, start=1):
            data = self._thin(panel_by_window[window])
            for coin in data.columns:
                series = data[coin].dropna()
                if series.empty:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=series.index, y=series.values, name=coin, legendgroup=coin,
                        showlegend=(row == 1),
                        line=dict(color=self._color(coin), width=1.4),
                        hovertemplate=f"<b>{coin}</b> %{{x|%Y-%m-%d %H:%M}}"
                                      "<br>%{y:.1f}% р.р.<extra></extra>",
                    ),
                    row=row, col=1,
                )
            if show_zero:
                fig.add_hline(y=0, line=dict(color=GRID_COLOR, width=1, dash="dot"),
                              row=row, col=1)
            if clip:
                fig.update_yaxes(range=[-clip, clip], row=row, col=1)
            fig.update_yaxes(title_text="% річних", ticksuffix="%", row=row, col=1)

        fig.update_layout(**layout(
            max(360, 260 * rows), margin=dict(l=60, r=20, t=48, b=40),
            legend=dict(orientation="h", y=1.06, x=0)))
        return fig

    # ------------------------------------------------------------------
    # ranking
    # ------------------------------------------------------------------
    def snapshot_heatmap(self, snapshot: pd.DataFrame) -> go.Figure:
        text = [[f"{v:+.0f}%" if pd.notna(v) else "" for v in row]
                for row in snapshot.values]
        fig = go.Figure(go.Heatmap(
            z=snapshot.values, x=list(snapshot.columns), y=list(snapshot.index),
            colorscale="RdYlGn", zmid=0, text=text, texttemplate="%{text}",
            hovertemplate="%{y} · %{x}: %{z:.1f}% р.р.<extra></extra>",
            colorbar=dict(title="% р.р."),
        ))
        fig.update_layout(**layout(
            60 + 34 * len(snapshot), hovermode="closest",
            margin=dict(l=60, r=10, t=10, b=10), yaxis=dict(autorange="reversed")))
        return fig

    def rank_history(self, ranks: pd.DataFrame) -> go.Figure:
        fig = go.Figure()
        thinned = self._thin(ranks)
        for coin in thinned.columns:
            series = thinned[coin].dropna()
            if series.empty:
                continue
            fig.add_trace(go.Scatter(
                x=series.index, y=series.values, name=coin, legendgroup=coin,
                line=dict(color=self._color(coin), width=1.2, shape="hv"),
                hovertemplate=f"<b>{coin}</b> %{{x|%Y-%m-%d}}: #%{{y:.0f}}<extra></extra>"))
        fig.update_layout(**layout(
            380, yaxis=dict(title="місце", autorange="reversed", dtick=1)))
        return fig

    def leader_shares(self, shares: pd.Series) -> go.Figure:
        fig = go.Figure(go.Bar(
            x=shares.index, y=shares.values,
            marker_color=[self._color(c) for c in shares.index],
            hovertemplate="%{x}: %{y:.1f}% часу<extra></extra>"))
        fig.update_layout(**layout(280, hovermode="closest", yaxis_ticksuffix="%",
                                   margin=dict(l=60, r=20, t=10, b=40)))
        return fig

    # ------------------------------------------------------------------
    # prices and equity
    # ------------------------------------------------------------------
    def price_lines(self, prices: pd.DataFrame, log_axis: bool, y_title: str) -> go.Figure:
        fig = go.Figure()
        thinned = self._thin(prices)
        for coin in thinned.columns:
            series = thinned[coin].dropna()
            if series.empty:
                continue
            fig.add_trace(go.Scatter(x=series.index, y=series.values, name=coin,
                                     legendgroup=coin,
                                     line=dict(color=self._color(coin), width=1.4)))
        fig.update_layout(**layout(
            520, yaxis=dict(type="log" if log_axis else "linear", title=y_title)))
        return fig

    def equity_curves(self, strategy: pd.Series, hold: pd.Series,
                      hold_label: str, log_axis: bool) -> go.Figure:
        strategy, hold = self._thin(strategy) * 100, self._thin(hold) * 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=strategy.index, y=strategy.values, name="Стратегія",
                                 line=dict(color=STRATEGY_COLOR, width=2)))
        fig.add_trace(go.Scatter(x=hold.index, y=hold.values, name=hold_label,
                                 line=dict(color=HOLD_COLOR, width=2, dash="dash")))
        fig.update_layout(**layout(380, yaxis=dict(
            title="капітал, % від старту", ticksuffix="%",
            type="log" if log_axis else "linear")))
        return fig

    def excess_area(self, excess_pct: pd.Series) -> go.Figure:
        excess_pct = self._thin(excess_pct)
        fig = go.Figure(go.Scatter(
            x=excess_pct.index, y=excess_pct.values, fill="tozeroy",
            line=dict(color=GAIN_COLOR, width=2), name="приріст понад базу"))
        fig.add_hline(y=0, line=dict(color=HOLD_COLOR, width=1, dash="dash"))
        fig.update_layout(**layout(280, yaxis=dict(title="приріст, %", ticksuffix="%")))
        return fig

    def drawdown_area(self, drawdown_pct: pd.Series) -> go.Figure:
        drawdown_pct = self._thin(drawdown_pct)
        fig = go.Figure(go.Scatter(
            x=drawdown_pct.index, y=drawdown_pct.values, fill="tozeroy",
            line=dict(color=LOSS_COLOR, width=1), name="просадка"))
        fig.update_layout(**layout(200, margin=dict(l=60, r=20, t=10, b=30),
                                   yaxis=dict(title="просадка, %", ticksuffix="%")))
        return fig

    def allocation_area(self, weights: pd.DataFrame) -> go.Figure:
        weights = self._thin(weights) * 100
        fig = go.Figure()
        for coin in weights.columns:
            fig.add_trace(go.Scatter(x=weights.index, y=weights[coin], name=coin,
                                     stackgroup="allocation",
                                     line=dict(width=0.5, color=self._color(coin))))
        fig.update_layout(**layout(320, margin=dict(l=60, r=20, t=10, b=30),
                                   yaxis=dict(title="% портфеля", range=[0, 100])))
        return fig


def styled_snapshot(table: pd.DataFrame):
    """Momentum table with a red-green gradient, falling back to plain numbers."""
    try:
        return table.style.format("{:+.1f}%").background_gradient(
            cmap="RdYlGn", vmin=-300, vmax=300)
    except ImportError:  # matplotlib not installed
        return table.round(1)


__all__ = ["ChartFactory", "PLOT_TEMPLATE", "styled_snapshot"]
