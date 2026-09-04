"""The five dashboard tabs, in the order they appear on screen."""
from .base import DashboardTab
from .curves import CurvesTab
from .dataset import DataTab
from .prices import PricesTab
from .ranking import RankingTab
from .simulation import SimulationTab

#: Order matters — this is the tab strip.
ALL_TABS: list[type[DashboardTab]] = [
    CurvesTab, RankingTab, PricesTab, SimulationTab, DataTab,
]

__all__ = ["ALL_TABS", "CurvesTab", "DashboardTab", "DataTab", "PricesTab",
           "RankingTab", "SimulationTab"]
