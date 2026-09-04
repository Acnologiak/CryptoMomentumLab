"""Tab 1 — the momentum curves themselves, one subplot per window."""
from __future__ import annotations

import streamlit as st

from ..context import DashboardContext
from .base import DashboardTab


class CurvesTab(DashboardTab):
    title = "📈 Криві моментуму"

    def render(self, ctx: DashboardContext) -> None:
        settings = ctx.settings
        curves = {w: ctx.in_range(ctx.panel[w]) for w in ctx.windows}
        figure = ctx.charts.momentum_curves(
            curves,
            show_zero=settings.show_zero,
            clip=settings.clip_limit if settings.clip_axis else None,
        )
        st.plotly_chart(figure, width="stretch")
        st.caption(
            "Значення — річна дохідність, до якої екстрапольовано рух за вікном "
            f"({settings.mode_caption}). Короткі вікна дають екстремальні числа — "
            "це нормально, порівнюйте монети між собою, а не з реальною річною "
            "дохідністю. Клік по легенді вимикає монету на всіх графіках."
        )
