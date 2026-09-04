"""Streamlit layer. Importing this pulls in streamlit, so the analysis
packages deliberately never depend on it."""
from .charts import ChartFactory
from .context import DashboardContext
from .dashboard import Dashboard
from .settings import AppSettings
from .sidebar import Sidebar
from .strategy_form import StrategyChoices, StrategyForm
from .strategy_report import StrategyReportView

__all__ = [
    "AppSettings", "ChartFactory", "Dashboard", "DashboardContext", "Sidebar",
    "StrategyChoices", "StrategyForm", "StrategyReportView",
]
