"""The control panel of the simulation tab.

Kept apart from the tab itself because it is almost entirely Streamlit widget
bookkeeping: bounds that move with the data, session-state clamping, and the
help texts that explain what each threshold does.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import streamlit as st

from ..analytics import MomentumMode, Smoothing
from ..config import (
    DEFAULT_SIM_ENTRY_EDGE,
    DEFAULT_SIM_POOL,
    DEFAULT_SIM_SLOTS,
    DEFAULT_SIM_WINDOW,
    WINDOW_CHOICES,
)
from ..strategy import SIZING_MODES, Sizing, StrategyConfig
from ..timeframe import window_label

SIZING_LABELS = {
    Sizing.INCREMENTAL.value: "Подієво: % від поточного BTC (без перебалансування)",
    Sizing.TARGET_WEIGHT.value: "Періодично: ціль 1/K портфеля",
}


@dataclass
class StrategyChoices:
    """What the user picked, before it becomes a validated `StrategyConfig`."""

    base: str
    pool: list[str]
    sizing: str
    window_days: float
    max_active: int
    rebalance_days: float
    entry_edge: float
    exit_edge: float
    fee_bps: float
    rotation_margin: float | None
    forecast_horizon_days: float | None
    slope_window_days: float
    fast_window_days: float | None

    def to_config(self, smoothing: Smoothing, smoothing_days: float,
                  mode: MomentumMode) -> StrategyConfig:
        """Momentum settings come from the sidebar so the simulation sees
        exactly the curves that are drawn in the other tabs."""
        return StrategyConfig(
            base=self.base,
            window_days=self.window_days,
            entry_edge=self.entry_edge,
            exit_edge=self.exit_edge,
            max_active=min(self.max_active, len(self.pool)),
            rebalance_days=self.rebalance_days,
            fee_bps=self.fee_bps,
            smoothing=smoothing,
            smoothing_days=smoothing_days,
            mode=mode,
            sizing=self.sizing,
            rotation_margin=self.rotation_margin,
            forecast_horizon_days=self.forecast_horizon_days,
            slope_window_days=self.slope_window_days,
            fast_window_days=self.fast_window_days,
        )


class StrategyForm:
    """Draws the simulation controls and collects them into `StrategyChoices`."""

    def __init__(self, base: str, alt_universe: Sequence[str]):
        self.base = base
        self.alt_universe = list(alt_universe)

    def render(self) -> StrategyChoices:
        sizing = self._sizing()
        window_days, max_active, rebalance_days = self._mechanics(sizing)
        entry_edge, exit_edge, fee_bps = self._thresholds()
        pool = self._pool()
        rotation_margin = self._rotation(sizing, entry_edge)
        forecast, slope_days, fast_window = self._early_exit()

        return StrategyChoices(
            base=self.base, pool=pool, sizing=sizing, window_days=window_days,
            max_active=max_active, rebalance_days=rebalance_days,
            entry_edge=entry_edge, exit_edge=exit_edge, fee_bps=fee_bps,
            rotation_margin=rotation_margin, forecast_horizon_days=forecast,
            slope_window_days=slope_days, fast_window_days=fast_window,
        )

    # ------------------------------------------------------------------
    def _sizing(self) -> str:
        return st.radio(
            "Механіка позиції", SIZING_MODES, index=SIZING_MODES.index("incremental"),
            horizontal=True, key="sim_sizing", format_func=SIZING_LABELS.get,
            help="«Подієво» — жодного розкладу: угода лише коли сигнал спрацював. "
                 "Новий вхід бере BTC_баланс / вільні_слоти (33% / 50% / 100% для "
                 "3 слотів), відкриті позиції ніхто не чіпає, вихід продає їх "
                 "повністю назад у BTC. «Періодично» — на кожному чекпоінті все "
                 "підганяється під фіксовану вагу 1/K від усього портфеля.")

    def _mechanics(self, sizing: str) -> tuple[float, int, float]:
        left, middle, right = st.columns(3)

        window_days = left.selectbox(
            "Вікно моментуму, днів", WINDOW_CHOICES,
            index=WINDOW_CHOICES.index(DEFAULT_SIM_WINDOW), key="sim_window",
            format_func=lambda d: f"{window_label(d)} ({d:g}д)")

        slot_max = len(self.alt_universe)
        slot_help = (
            "Скільки альтів можуть тримати позицію одночасно; вхід ділить поточний "
            "BTC на 1/(вільні слоти)." if sizing == Sizing.INCREMENTAL.value else
            "Кожен активний альт тримає фіксовану частку 1/K портфеля.")
        max_active = middle.slider(
            "Активних слотів під альти, K", 1, slot_max, key="sim_k", help=slot_help,
            **self._clamped("sim_k", slot_max, min(DEFAULT_SIM_SLOTS, slot_max)))

        event_driven = sizing == Sizing.INCREMENTAL.value
        rebalance_days = right.number_input(
            "Перевірка/ребаланс, днів", 0.25, 30.0, 1.0, 0.25, key="sim_rebal",
            disabled=event_driven,
            help="Ігнорується в подієвому режимі — там перевірка щобару."
                 if event_driven else
                 "Як часто стратегія перевіряє сигнал і підганяє ваги під ціль.")
        return window_days, max_active, rebalance_days

    def _thresholds(self) -> tuple[float, float, float]:
        left, middle, right = st.columns(3)

        entry_edge = left.number_input(
            "Поріг входу, п.п. річних", 0.0, 2000.0, DEFAULT_SIM_ENTRY_EDGE, 5.0,
            key="sim_buy",
            help="Наскільки моментум альта має обійти BTC, щоб відкрити позицію.")

        exit_edge = middle.number_input(
            "Поріг виходу («схожості»), п.п. річних", min_value=0.0,
            max_value=float(entry_edge), step=1.0, key="sim_exit",
            help="Коли перевага альта над BTC падає нижче цього — повернення в BTC.",
            **self._clamped("sim_exit", float(entry_edge), min(0.0, float(entry_edge))))

        fee_bps = right.number_input(
            "Комісія за угоду, б.п. від обсягу", 0.0, 200.0, 10.0, 5.0, key="sim_fee",
            help="10 б.п. = 0.10% від обсягу кожної ноги угоди.")
        return entry_edge, exit_edge, fee_bps

    def _pool(self) -> list[str]:
        preset = [c for c in DEFAULT_SIM_POOL if c in self.alt_universe]
        return st.multiselect(
            "Пул альткоїнів для ротації", self.alt_universe,
            default=preset or self.alt_universe, key="sim_pool",
            help="Ширший пул за K слотів + ротація нижче = стратегія сама шукає "
                 "найсильніший моментум серед усіх, не лише серед перших K обраних. "
                 "Обережно з розширенням: єдиний набір, стійкість якого підтверджена "
                 "на train+test розколі, — це ETH/SOL/BNB. Ширші пули різко "
                 "піднімають дохідність на бектесті, але це майже завжди заслуга "
                 "однієї монети з різким разовим ривком, а не системна перевага "
                 "(перевірено на 648 комбінаціях параметрів).")

    def _rotation(self, sizing: str, entry_edge: float) -> float | None:
        left, right = st.columns(2)
        event_driven = sizing == Sizing.INCREMENTAL.value

        enabled = left.checkbox(
            "Ротація слотів (заміна найслабшого сильнішим ззовні)", value=False,
            key="sim_rotation_on", disabled=not event_driven,
            help="Якщо в пулі більше монет, ніж слотів K: без ротації вже відкрита "
                 "позиція тримається, доки не впаде нижче порогу виходу сама по "
                 "собі — навіть якщо десь поза слотами є набагато сильніший "
                 "моментум. З ротацією такий сильніший кандидат витісняє найслабший "
                 "слот (продаж у BTC → купівля кандидата з BTC).")

        margin = right.number_input(
            "Поріг ротації, п.п. річних", min_value=0.0, max_value=2000.0, step=1.0,
            key="sim_rotation_margin", disabled=not enabled or not event_driven,
            help="Наскільки перевага кандидата над BTC має перевищити перевагу "
                 "найслабшого поточного слота, щоб відбулась заміна (проти "
                 "тремтіння на майже рівних монетах).",
            **self._clamped("sim_rotation_margin", 2000.0, min(10.0, float(entry_edge))))

        return margin if (enabled and event_driven) else None

    def _early_exit(self) -> tuple[float | None, float, float | None]:
        with st.expander("⏱ Ранній вихід у BTC — нівелювання затримки моментуму"):
            st.caption(
                "Обидва механізми лише **прискорюють** повернення в BTC: вони ніколи "
                "не блокують і не відкладають вхід і ніколи не тримають позицію "
                "довше. Вимкнені за замовчуванням — поведінка тоді точно як без них.")

            one, two, three = st.columns(3)
            forecast_on = one.checkbox(
                "Прогноз перетину", value=False, key="sim_forecast_on",
                help="Провести МНК-пряму через останні N днів переваги альта над BTC "
                     "і продовжити її. Якщо ця пряма перетне поріг виходу протягом "
                     "горизонту (і нахил від'ємний) — повернення в BTC зараз, не "
                     "чекаючи фактичного перетину. Це похідний член на правилі виходу.")
            horizon = two.number_input(
                "Горизонт прогнозу, днів", 0.5, 60.0, 7.0, 0.5, key="sim_forecast_days",
                disabled=not forecast_on,
                help="Наскільки далеко вперед дивиться екстраполяція. Орієнтир — "
                     "сумарна затримка згладжування плюс приблизно пів-вікна моментуму.")
            slope_days = three.number_input(
                "Вікно нахилу, днів", 1.0, 60.0, 5.0, 0.5, key="sim_slope_days",
                disabled=not forecast_on,
                help="За скільки останніх днів рахується нахил прямої. Менше — "
                     "швидша реакція й більше шуму в оцінці; більше — плавніше, "
                     "але сам нахил запізнюється.")

            left, right = st.columns(2)
            veto_on = left.checkbox(
                "Вето швидкого вікна", value=False, key="sim_fast_on",
                help="Перерахувати перевагу над BTC на коротшому вікні моментуму. "
                     "Якщо на ньому альт уже втратив перевагу (або власний момент "
                     "став від'ємним) — вихід, навіть якщо повільне вікно ще тримає. "
                     "Менше лагу — раніше ловить розворот, але й більше тіпань на шумі.")
            fast_window = right.selectbox(
                "Швидке вікно, днів", WINDOW_CHOICES, index=WINDOW_CHOICES.index(7.0),
                key="sim_fast_window", disabled=not veto_on,
                format_func=lambda d: f"{window_label(d)} ({d:g}д)")

        return (float(horizon) if forecast_on else None,
                float(slope_days),
                float(fast_window) if veto_on else None)

    # ------------------------------------------------------------------
    @staticmethod
    def _clamped(key: str, ceiling: float, first_value):
        """Keep a stored widget value inside bounds that move with the data.

        Streamlit warns if you write to `session_state` for a key *and* pass
        that widget's `value=` in the same run, so the stored value is clamped
        here and `value=` is only supplied the very first time the widget is
        created (when there is nothing in session state yet).
        """
        if key in st.session_state:
            if st.session_state[key] > ceiling:
                st.session_state[key] = ceiling
            return {}
        return {"value": first_value}
