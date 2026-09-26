"""Согласованная загрузка данных для обеих таблиц статьи.

Смысл этого модуля — чтобы все семейства моделей считались на одном и том же
входе. Если у деревьев панель из 124 рядов с 1971 года, а у линейных моделей из
126 с 1979-го, разница в R² между семействами будет включать разницу в данных.

Сценарий этой ветки: НЕПОЛНЫЕ РЯДЫ ОТБРАСЫВАЮТСЯ, ВСЕ СТРОКИ СОХРАНЯЮТСЯ.

Три ряда FRED-MD начинаются позже начала выборки: ACOGNO с 1992-03, UMCSENTx с
1978-02, TWEXMMTH с 1973-02. Здесь они убираются как колонки, и история остаётся
полной с 1971-08. Цена — 3 предиктора из 127, то есть 2.4%.

Альтернатива (ветка macro-drop-rows) убирает строки вместо колонок. Тогда выборка
начинается с 1978-02, и на первом прогнозе в 1989-01 обучающая история сжимается
с 209 месяцев до 131, то есть на 37%. В независимых годовых наблюдениях это 17
против 11, а расширяющееся окно и без того голодает на первых прогнозах.

Рваный край в конце панели (12 рядов без значений за 2018-11, один за 2018-10 —
обычные лаги публикации) значения не имеет: последняя дата прогноза — 2017-12,
потому что доходность с неё реализуется в 2018-12.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data_processed"

MATURITIES = [2, 3, 4, 5, 7, 10]
LAST_ORIGIN = pd.Period("2017-12", freq="M")
INCOMPLETE_SERIES = ["ACOGNO", "UMCSENTx", "TWEXMMTH"]


def _read(name):
    frame = pd.read_csv(PROCESSED_DIR / name, index_col="date")
    frame.index = pd.PeriodIndex(frame.index, freq="M")
    return frame.sort_index()


def load():
    """Возвращает (forwards, macro, excess) на общем индексе дат прогноза.

    Доходности в ПРОЦЕНТАХ, как в исходных файлах. Это соглашение важно соблюдать
    во всех семействах моделей: elastic net принципиально зависит от единиц
    измерения цели. В его функции потерь при умножении цели на 100 квадратичная
    ошибка и L2-штраф растут в 10 000 раз, а L1-штраф только в 100, поэтому одним
    множителем alpha их не согласовать. На наших данных переход с процентов на
    доли сдвигает R² elastic net на 0.9-3.8 п.п. Ridge и lasso инвариантны.

    Строка датирована моментом прогноза t, а доходность в excess реализуется в t+12.
    """
    forwards = _read("forward_rates.csv")
    excess = _read("excess_returns.csv")[[f"xr_{n}y" for n in MATURITIES]]
    macro = _read("macro_panel.csv")

    macro = macro.drop(columns=[c for c in INCOMPLETE_SERIES if c in macro.columns])

    common = forwards.index.intersection(macro.index).intersection(excess.index)
    common = common[common <= LAST_ORIGIN]
    return forwards.loc[common], macro.loc[common], excess.loc[common]


if __name__ == "__main__":
    forwards, macro, excess = load()
    remaining = macro.isna().sum().sum()
    print(f"forwards: {forwards.shape}  macro: {macro.shape}  excess: {excess.shape}")
    print(f"окно: {forwards.index[0]} .. {forwards.index[-1]}  ({len(forwards)} дат прогноза)")
    print(f"пропусков в макропанели: {remaining}")
