"""Сеть с одним выходом: предсказываем общий фактор премии, сроки получаем через нагрузки.

Нагрузки b_n оцениваются регрессией rx_n на фактор внутри того же обучающего окна,
так что никакой информации из будущего не добавляется.
"""
import sys, time
import numpy as np
import pandas as pd
import NN_1layer_3nodes as m

STANDARDIZE_Y = len(sys.argv) > 1 and sys.argv[1] == "std"
m.STANDARDIZE_Y = STANDARDIZE_Y

X, Y = m.load_data()
mats = list(Y.columns)
origins = X.index[X.index >= m.FIRST_ORIGIN]

actual = np.full((len(origins), len(mats)), np.nan)
forecast = np.full_like(actual, np.nan)
benchmark = np.full_like(actual, np.nan)

hyper = dict(dropout=m.NN_CFG["dropout"], weight_decay=m.NN_CFG["weight_decay"])
last_tune, t0 = -m.TUNE_EVERY, time.time()

for i, origin in enumerate(origins):
    hist = X.index <= origin - m.GAP
    X_hist, Y_hist = X[hist].values, Y[hist].values
    factor_hist = Y_hist.mean(axis=1, keepdims=True)

    if i - last_tune >= m.TUNE_EVERY:
        best = m.tune_hyperparameters(X_hist, factor_hist, m.NN_CFG, m.HYPER_GRID)
        if best is not None:
            hyper = best
        last_tune = i
        print(f"    [tune] t={origin} -> {hyper}", flush=True)

    ens, sc_x, y_mean, y_std, _ = m.train_ensemble(
        X_hist, factor_hist, m.NN_CFG, hyper,
        n_seeds=m.NN_CFG["n_seeds"], top_k=m.NN_CFG["top_k"])
    factor_hat = float(m.predict_ensemble(ens, X.loc[[origin]].values, sc_x, y_mean, y_std)[0, 0])

    # нагрузки: rx_n = a_n + b_n * factor, оценка по истории
    design = np.column_stack([np.ones(len(factor_hist)), factor_hist.ravel()])
    coef, *_ = np.linalg.lstsq(design, Y_hist, rcond=None)

    forecast[i, :] = coef[0] + coef[1] * factor_hat
    actual[i, :] = Y.loc[origin].values
    benchmark[i, :] = Y_hist.mean(axis=0)

    if (i + 1) % 60 == 0:
        el = time.time() - t0
        print(f"    [{i+1}/{len(origins)}] elapsed={el:.0f}s ETA={el/(i+1)*(len(origins)-i-1):.0f}s", flush=True)

df = pd.DataFrame(index=origins)
for j, c in enumerate(mats):
    df[f"actual_{c}"], df[f"forecast_{c}"], df[f"benchmark_{c}"] = actual[:, j], forecast[:, j], benchmark[:, j]

print(f"\n### ОДИН ВЫХОД, STANDARDIZE_Y={STANDARDIZE_Y} ###")
m.report(df, mats)
for c in mats:
    print(f"  корреляция {c}: {np.corrcoef(df[f'actual_{c}'], df[f'forecast_{c}'])[0,1]:.2f}")
df.to_csv(f"single_output_{'std' if STANDARDIZE_Y else 'raw'}.csv")
print(f"время {time.time()-t0:.0f}s")
