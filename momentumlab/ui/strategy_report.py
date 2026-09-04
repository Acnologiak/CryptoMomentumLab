"""Rendering a finished backtest: headline numbers, charts, journals."""
from __future__ import annotations

import streamlit as st

from ..strategy import (
    JUSTIFIED,
    BacktestResult,
    ExitTrigger,
    PerformanceReport,
    Sizing,
)
from .charts import ChartFactory

EARLY_EXIT_HEADERS = {
    "time": "час",
    "coin": "монета",
    "trigger": "тригер",
    "diff": "перевага, п.п.",
    "predicted_days": "прогноз, дн.",
    "verdict": "вердикт",
    "slow_exit_after_days": "повільний вихід через, дн.",
}

SIZING_NOTES = {
    Sizing.INCREMENTAL: (
        "Подієвий режим: жодного розкладу — перевірка щобару, торгується лише коли "
        "сигнал змінився. Новий вхід бере **BTC_баланс / вільні_слоти** (для K=3: "
        "33% на перший вхід, 50% від того, що лишилось, на другий, решта — на "
        "третій); уже відкриті позиції ніхто не чіпає й не перебалансовує."),
    Sizing.TARGET_WEIGHT: (
        "Періодичний режим: на кожному чекпоінті (з інтервалом ребалансу) всі "
        "активні позиції підганяються рівно під 1/K портфеля, навіть якщо вже були "
        "відкриті — це і є джерело значно більшої кількості угод/комісій порівняно "
        "з подієвим режимом."),
}


class StrategyReportView:
    """Draws everything the simulation tab shows about one result."""

    def __init__(self, result: BacktestResult, charts: ChartFactory):
        self.result = result
        self.report = PerformanceReport(result)
        self.charts = charts
        self.summary = self.report.summary()

    def render(self) -> None:
        self._headline()
        self._equity()
        self._excess()
        self._drawdown()
        st.dataframe(self.summary.round(2), width="stretch")
        self._allocation()
        self._trade_journal()
        self._early_exit_journal()
        self._footnote()

    # ------------------------------------------------------------------
    def _headline(self) -> None:
        base = self.result.base
        hold_row = f"Утримання {base}"
        strategy_cagr = self.summary.loc["Стратегія", "CAGR, %"]
        hold_cagr = self.summary.loc[hold_row, "CAGR, %"]
        excess = self.result.excess_multiple

        cells = st.columns(4)
        # "x times over BTC" is strategy capital / buy-and-hold capital over the
        # whole period — the net contribution of the rotation, with the part of
        # the move both legs share divided out.
        cells[0].metric(f"Приріст понад {base}", f"×{excess:.2f}",
                        f"{(excess - 1) * 100:+.1f}% за весь період")
        cells[1].metric(f"CAGR: стратегія vs {base}", f"{strategy_cagr:+.1f}%",
                        f"{strategy_cagr - hold_cagr:+.1f} п.п./рік понад {base}")
        cells[2].metric("Угод", int(self.summary.loc["Стратегія", "угод"]))
        cells[3].metric(
            "Комісій сплачено",
            f"{self.summary.loc['Стратегія', 'комісій, % від капіталу']:.1f}% капіталу")

        st.caption(
            f"Для довідки: дохідність стратегії "
            f"{self.summary.loc['Стратегія', 'дохідність, %']:+.1f}% проти утримання "
            f"{base} {self.summary.loc[hold_row, 'дохідність, %']:+.1f}% "
            f"за цей же період.")

    def _equity(self) -> None:
        log_axis = st.checkbox("Лог-шкала капіталу", True, key="sim_logy")
        st.plotly_chart(
            self.charts.equity_curves(self.result.equity, self.result.hold_equity,
                                      f"Утримання {self.result.base}", log_axis),
            width="stretch")

    def _excess(self) -> None:
        base = self.result.base
        st.subheader(f"Чистий приріст понад просте тримання {base}")
        st.caption(
            f"Капітал стратегії поділений на капітал {base}-холду — прибирає "
            "спільний рух ринку і показує лише те, що додає сама ротація. Пряма "
            "лінія на нулі означає «стратегія = " + base + "», нижче нуля — "
            "стратегія відстає.")
        excess_pct = (self.result.excess_over_base - 1.0) * 100
        st.plotly_chart(self.charts.excess_area(excess_pct), width="stretch")

    def _drawdown(self) -> None:
        st.plotly_chart(self.charts.drawdown_area(self.report.drawdown() * 100),
                        width="stretch")

    def _allocation(self) -> None:
        st.subheader("Розподіл портфеля з часом")
        columns = [self.result.base] + self.result.alts
        st.plotly_chart(self.charts.allocation_area(self.result.weights[columns]),
                        width="stretch")

        activity = self.report.coin_activity()
        if not activity.empty:
            st.caption("Частка ребалансів, де альт тримав слот: " +
                       "; ".join(f"**{c}** {v:.0f}%" for c, v in activity.items()))

    # ------------------------------------------------------------------
    def _trade_journal(self) -> None:
        trades = self.result.trades
        with st.expander(f"Журнал угод ({len(trades)})"):
            if trades.empty:
                st.write("Угод не було — жоден альт не пробив поріг входу "
                         "за обраний період.")
                return
            newest_first = trades.sort_values("time", ascending=False)
            numeric = newest_first.select_dtypes("number").columns
            st.dataframe(newest_first.round({c: 4 for c in numeric}), width="stretch")
            st.download_button("⬇ CSV: журнал угод",
                               trades.to_csv(index=False).encode(),
                               file_name="backtest_trades.csv", mime="text/csv")

    def _early_exit_journal(self) -> None:
        if not self.result.config.has_early_exit:
            return

        scored = self.report.early_exits()
        with st.expander(f"Ранні виходи в BTC ({len(scored)})"):
            if scored.empty:
                st.write("Жодного раннього виходу — прискорювачі увімкнені, "
                         "але жоден не спрацював за цей період.")
                return

            total = len(scored)
            justified = int((scored["verdict"] == JUSTIFIED).sum())
            forecast = int((scored["trigger"] == ExitTrigger.FORECAST.value).sum())
            st.caption(
                f"**{justified}/{total}** виправдані (повільне правило все одно "
                f"вийшло б у межах горизонту), **{total - justified}** передчасні "
                f"(перевага відновилась). Тригер: «{ExitTrigger.FORECAST.value}» — "
                f"{forecast}, «{ExitTrigger.FAST_WINDOW.value}» — {total - forecast}.")

            shown = scored.assign(
                time=scored["time"].dt.strftime("%Y-%m-%d %H:%M"),
                diff=scored["diff"].round(1),
                predicted_days=scored["predicted_days"].round(1),
                slow_exit_after_days=scored["slow_exit_after_days"].round(1),
            ).rename(columns=EARLY_EXIT_HEADERS)
            st.dataframe(shown, width="stretch", hide_index=True)

    def _footnote(self) -> None:
        config = self.result.config
        st.caption(
            "Гістерезис: альт отримує слот, коли його моментум перевищує "
            f"{config.base} більш ніж на **поріг входу** (і сам по собі додатний); "
            "слот звільняється й портфель повертається "
            f"в {config.base}, коли перевага падає нижче **порогу виходу** "
            "(«практично дорівнює BTC») або момент альта стає від'ємним. Якщо "
            f"кандидатів на слот більше за K, лишаються ті, у кого перевага над "
            f"{config.base} найбільша. {SIZING_NOTES[config.sizing]} Угода "
            "виконується по ціні закриття того самого бару без прослизання; "
            f"комісія береться за кожну ногу {config.base}↔альт, тож фактична "
            "заміна альт→альт коштує двох комісій. Прискорювачі виходу (розділ "
            f"«Ранній вихід у {config.base}») лише наближають повернення в "
            f"{config.base} — на вхід не впливають. Це дослідницький інструмент, "
            "а не інвестиційна порада.")
