"""Загрузка данных из тех же файлов, которые используют остальные модели.

Читается `yield_only_data.csv` и `macro_data.csv` — финальные матрицы, а не
промежуточный `macro_panel.csv`. Смысл в том, чтобы все три семейства моделей
считались на одинаковом входе: иначе разница в R² между линейными моделями,
деревьями и сетями будет частично отражать разницу в данных, а не в методах.

Оба файла датированы моментом РЕАЛИЗАЦИИ доходности, поэтому здесь индекс
сдвигается на 12 месяцев назад — к дате прогноза. Так строка с датой t содержит
предикторы, известные в t, и доходность, которая реализуется в t+12.

Доходности в ПРОЦЕНТАХ, как в исходных файлах. Это соглашение важно соблюдать во
всех семействах моделей: elastic net принципиально зависит от единиц измерения
цели. При умножении цели на 100 квадратичная ошибка и L2-штраф растут в 10 000
раз, а L1-штраф только в 100, поэтому одним множителем alpha их не согласовать.
На наших данных переход с процентов на доли сдвигает R² elastic net на 0.9-3.8
п.п. Ridge и lasso инвариантны.

Замечание о длине истории. `macro_data.csv` начинается с 1979-02 (по дате
реализации), потому что при его сборке строки с пропусками были отброшены. На
первом прогнозе в январе 1989 обучающей истории поэтому 131 месяц вместо 209.
Это не ошибка: заглядывания вперёд нет, просто меньше данных. На тестовую выборку
из 348 прогнозов это не влияет. Альтернатива — отбрасывать неполные ряды как
колонки, а не строки, тогда история сохраняется с 1971-08; см. историю PR #1 и #2.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data_processed"

MATURITIES = [2, 3, 4, 5, 7, 10]
HORIZON = 12
TARGETS = [f"xr_{n}y" for n in MATURITIES]


def _read(name):
    frame = pd.read_csv(PROCESSED_DIR / name, index_col="date")
    frame.index = pd.PeriodIndex(frame.index, freq="M") - HORIZON
    return frame.sort_index()


def load():
    """Возвращает (forwards, macro, excess) на общем индексе дат прогноза."""
    yield_only = _read("yield_only_data.csv")
    macro_full = _read("macro_data.csv")

    forwards = yield_only.drop(columns=TARGETS)
    excess = yield_only[TARGETS]
    macro = macro_full.drop(columns=TARGETS + list(forwards.columns))

    common = forwards.index.intersection(macro.index)
    return forwards, macro.loc[common], excess


if __name__ == "__main__":
    forwards, macro, excess = load()
    print(f"forwards: {forwards.shape}  macro: {macro.shape}  excess: {excess.shape}")
    print(f"форварды и цели: {forwards.index[0]} .. {forwards.index[-1]}")
    print(f"макро:           {macro.index[0]} .. {macro.index[-1]}")
    print(f"пропусков: forwards {forwards.isna().sum().sum()}, "
          f"macro {macro.isna().sum().sum()}, excess {excess.isna().sum().sum()}")
