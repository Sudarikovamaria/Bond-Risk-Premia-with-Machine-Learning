"""Таблица 1, панели A и B: прогноз избыточных доходностей по форвардным ставкам.

Первый прогноз делается в 1989-01 и реализуется в 1990-01, последний — в 2018-12.
Фраза статьи "recursive forecast which starts in January 1990" относится к дате
реализации: в §4.1 сказано, что первая ошибка прогноза сравнивает доходность за
февраль 1989 - январь 1990 с прогнозом, сделанным в январе 1989.

GAP=1 воспроизводит статью, GAP=12 убирает заглядывание вперёд: годовая доходность
с датой t становится известна только в t+12.

Запуск:  python models/PCA_PLS_linear_models/run_table1.py
"""

from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import clark_west, forecast, r2_oos
from data_loading import MATURITIES, load
from linear_models import (
    ENET_GRID,
    LASSO_GRID,
    RIDGE_GRID,
    elastic_net,
    lasso,
    pcr,
    pls,
    ridge,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "micro"
FIRST_ORIGIN = pd.Period("1989-01", freq="M")
GAP = 1

SPECIFICATIONS = [
    ("PCA (10 components)", partial(pcr, n_components=10), None),
    ("PCA (5 components)", partial(pcr, n_components=5), None),
    ("PCA (3 components)", partial(pcr, n_components=3), None),
    ("PCA-Squared (5 components)", partial(pcr, n_components=5, squared=True), None),
    ("PCA-Squared (3 components)", partial(pcr, n_components=3, squared=True), None),
    ("Partial Least Squares (5 components)", partial(pls, n_components=5), None),
    ("Partial Least Squares (3 components)", partial(pls, n_components=3), None),
    ("Ridge", ridge, RIDGE_GRID),
    ("Lasso", lasso, LASSO_GRID),
    ("Elastic Net", elastic_net, ENET_GRID),
]


def run_specification(name, predict, grid, X, excess):
    row = {"model": name}
    predictions = {}

    for n in MATURITIES:
        result, _ = forecast(
            X, excess[f"xr_{n}y"], predict, grid=grid, gap=GAP, first_origin=FIRST_ORIGIN
        )
        predictions[n] = result
        row[f"r2_{n}"] = r2_oos(result.actual, result.forecast, result.benchmark)
        row[f"p_{n}"] = clark_west(result.actual, result.forecast, result.benchmark)[1]

    portfolio = sum(predictions[n] for n in MATURITIES) / len(MATURITIES)
    row["r2_ew"] = r2_oos(portfolio.actual, portfolio.forecast, portfolio.benchmark)
    row["p_ew"] = clark_west(portfolio.actual, portfolio.forecast, portfolio.benchmark)[1]
    return row, len(portfolio)


def format_table(results):
    out = pd.DataFrame(index=results["model"])
    for key in [str(n) for n in MATURITIES] + ["ew"]:
        r2 = results[f"r2_{key}"].values
        p = results[f"p_{key}"].values
        out[f"xr_{key}"] = [f"{100 * v:.1f}%" for v in r2]
        out[f"p_{key}"] = [
            f"{pv:.3f}" if r2v > 0 and np.isfinite(pv) else "" for r2v, pv in zip(r2, p)
        ]
    return out


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    forwards, _, excess = load()

    rows, n_obs = [], None
    for name, predict, grid in SPECIFICATIONS:
        row, n_obs = run_specification(name, predict, grid, forwards, excess)
        rows.append(row)
        print(
            f"{name:38s} "
            + " ".join(f"{100 * row[f'r2_{n}']:6.1f}%" for n in MATURITIES)
            + f"  EW {100 * row['r2_ew']:6.1f}%",
            flush=True,
        )

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_DIR / f"table1_raw_gap{GAP}.csv", index=False)
    format_table(results).to_csv(RESULTS_DIR / f"table1_gap{GAP}.csv")
    print(f"\nпрогнозов вне выборки: {n_obs}")


if __name__ == "__main__":
    main()
