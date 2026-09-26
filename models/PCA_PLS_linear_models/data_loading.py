"""Согласованная загрузка данных для обеих таблиц статьи.

Смысл этого модуля — чтобы все семейства моделей считались на одном и том же
входе. Если у деревьев панель из 124 рядов с 1971 года, а у линейных моделей из
126 с 1979-го, разница в R² между семействами будет включать разницу в данных.

Сценарий этой ветки: ВСЕ РЯДЫ СОХРАНЯЮТСЯ, СТРОКИ С ПРОПУСКАМИ ОТБРАСЫВАЮТСЯ.

Три ряда FRED-MD начинаются позже начала выборки: ACOGNO с 1992-03, UMCSENTx с
1978-02, TWEXMMTH с 1973-02. Здесь сохраняются все 127 предикторов, а выборка
начинается с того месяца, когда доступны все они, то есть с 1992-03.

Цена высокая: на первом прогнозе в 1989-01 обучающей истории не остаётся вообще,
поэтому прогнозы могут начаться только заметно позже 1990 года, вопреки статье.
Альтернатива (ветка macro-drop-columns) убирает три колонки вместо строк и
сохраняет полную историю с 1971-08 при 124 предикторах.

Рваный край в конце панели (12 рядов без значений за 2018-11, один за 2018-10 —
обычные лаги публикации) отсекается отдельно: последняя дата прогноза — 2017-12,
потому что доходность с неё реализуется в 2018-12. Иначе dropna съел бы и эти
строки за компанию.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data_processed"

MATURITIES = [2, 3, 4, 5, 7, 10]
LAST_ORIGIN = pd.Period("2017-12", freq="M")


def _read(name):
    frame = pd.read_csv(PROCESSED_DIR / name, index_col="date")
    frame.index = pd.PeriodIndex(frame.index, freq="M")
    return frame.sort_index()


def load():
    """Возвращает (forwards, macro, excess) на общем индексе дат прогноза.

    Доходности в процентах, как в исходных файлах. Строка датирована моментом
    прогноза t, а доходность в excess реализуется в t+12.
    """
    forwards = _read("forward_rates.csv")
    excess = _read("excess_returns.csv")[[f"xr_{n}y" for n in MATURITIES]]
    macro = _read("macro_panel.csv")

    macro = macro.loc[:LAST_ORIGIN].dropna(axis=0, how="any")

    common = forwards.index.intersection(macro.index).intersection(excess.index)
    common = common[common <= LAST_ORIGIN]
    return forwards.loc[common], macro.loc[common], excess.loc[common]


if __name__ == "__main__":
    forwards, macro, excess = load()
    print(f"forwards: {forwards.shape}  macro: {macro.shape}  excess: {excess.shape}")
    print(f"окно: {forwards.index[0]} .. {forwards.index[-1]}  ({len(forwards)} дат прогноза)")
    print(f"пропусков в макропанели: {macro.isna().sum().sum()}")
