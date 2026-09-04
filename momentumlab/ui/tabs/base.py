"""Shared shape of a dashboard tab."""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..context import DashboardContext


class DashboardTab(ABC):
    """One tab: a label for the tab strip and everything drawn inside it."""

    #: text shown on the tab strip, emoji included
    title: str = ""

    @abstractmethod
    def render(self, ctx: DashboardContext) -> None:
        """Draw the tab. Called inside the tab's own Streamlit container."""
