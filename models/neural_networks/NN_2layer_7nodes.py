"""
Архитектура: NN 2 Layer (7 nodes each), 6 outputs
"""

import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

try:
    torch.set_num_threads(2)
except RuntimeError:
    pass


ROOT = Path(__file__).resolve().parent
LW_FILE = ROOT / "LW_monthly.xlsx"
MD_FILE = ROOT / "FRED-MD_2018m12.csv"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SAMPLE_START = "1971-08"
SAMPLE_END   = "2018-12"
HOLDING      = 12

FIRST_ORIGIN = pd.Period("1989-01", freq="M")
GAP          = 1
MATURITIES   = [2, 3, 4, 5, 7, 10]
CW_LAGS      = 11

FORWARD_HORIZONS = list(range(2, 11))

STANDARDIZE_Y = False

NN_CFG = dict(
    hidden=(7, 7),       # NN 2 Layer (7 nodes each)
    dropout=0.1,
    weight_decay=1e-4,
    lr=1e-2,
    n_seeds=100,
    top_k=10,
    epochs=1000,
    patience=20,
    batch_size=32,
    train_frac=0.85,
)

HYPER_GRID = [
    dict(dropout=0.1, weight_decay=1e-3),
    dict(dropout=0.1, weight_decay=1e-4),
    dict(dropout=0.3, weight_decay=1e-3),
    dict(dropout=0.3, weight_decay=1e-4),
    dict(dropout=0.5, weight_decay=1e-3),
    dict(dropout=0.5, weight_decay=1e-4),
]
TUNE_EVERY = 60
TUNE_SEEDS = 5

_MONTH_RE = re.compile(r"^\s*(\d+)\s*[_ ]?\s*m\s*$", re.I)


def _parse_months(col):
    m = _MONTH_RE.match(str(col))
    return int(m.group(1)) if m else None


def _to_yyyymm(v):
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return int(v)
    return int(str(v).strip().split(".")[0])


def load_liu_wu(path: Path) -> pd.DataFrame:
    print(f"[data] Читаю {path.name}...")
    ext = path.suffix.lower()
    if ext in (".xlsx", ".xls"):
        raw = pd.read_excel(path, header=None, dtype=str)
    else:
        raw = pd.read_csv(path, header=None, dtype=str)

    header_row = None
    for i in range(min(30, len(raw))):
        row = raw.iloc[i].astype(str).tolist()
        if sum(_MONTH_RE.match(str(x)) is not None for x in row) >= 5:
            header_row = i
            break
    if header_row is None:
        raise ValueError("Liu-Wu: заголовки не найдены")

    header = raw.iloc[header_row].tolist()
    header[0] = "date"
    data = raw.iloc[header_row + 1:].copy()
    data.columns = header
    data = data.reset_index(drop=True)

    data["date"] = data["date"].map(_to_yyyymm)
    data["date"] = pd.PeriodIndex(data["date"].astype(str), freq="M")
    data = data.set_index("date").sort_index()

    mat_cols = [c for c in data.columns if _parse_months(c) is not None]
    data = data[mat_cols]

    def _to_num(x):
        if isinstance(x, str):
            x = x.replace(",", ".")
        return pd.to_numeric(x, errors="coerce")
    data = data.applymap(_to_num)

    data.columns = [_parse_months(c) / 12.0 for c in data.columns]
    data = data[sorted(data.columns)]

    if data.abs().max().max() > 1.0:
        data = data / 100.0
    print(f"  shape={data.shape}, {data.index[0]}..{data.index[-1]}")
    return data


def load_fred_md(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, header=None, dtype=str)
    header = raw.iloc[0].tolist()
    header[0] = "date"
    data = raw.iloc[2:].copy()
    data.columns = header
    data = data.reset_index(drop=True)
    data["date"] = pd.PeriodIndex(pd.to_datetime(data["date"]), freq="M")
    return data.set_index("date").sort_index()


def build_forward_rates(y: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=y.index)
    out["short_rate"] = y[1.0]
    for n in FORWARD_HORIZONS:
        cur, prev = y[float(n)], y[float(n - 1)]
        out[f"fwd_{n}y"] = n * cur - (n - 1) * prev
    return out


def build_excess_returns(y: pd.DataFrame, mats=MATURITIES, holding=HOLDING):
    out = pd.DataFrame(index=y.index)
    for n in mats:
        y_n, y_n1, y_1 = y[float(n)], y[float(n - 1)], y[1.0]
        future = y_n1.shift(-holding)
        out[f"xr_{n}y"] = -(n - 1) * (future - y_n) + (y_n - y_1)
    return out


def load_data():
    print("Данные")
    lw = load_liu_wu(LW_FILE)
    lo = pd.Period(SAMPLE_START, freq="M")
    hi = pd.Period(SAMPLE_END, freq="M")
    lw = lw[(lw.index >= lo) & (lw.index <= hi)]

    fwd = build_forward_rates(lw)
    exc = build_excess_returns(lw, MATURITIES, HOLDING)
    md = load_fred_md(MD_FILE)

    common = fwd.index.intersection(exc.dropna().index).intersection(md.index)
    common = common[(common >= lo) & (common <= hi)]

    X = fwd.loc[common]
    Y = exc.loc[common]
    print(f"X={X.shape}, Y={Y.shape}, {common[0]}..{common[-1]}")
    print(f"Y stats: mean={Y.values.mean():.5f}, std={Y.values.std():.5f}")
    for c in Y.columns:
        print(f"  {c}: mean={Y[c].mean():+.5f}, std={Y[c].std():.5f}")
    return X, Y

class MLP(nn.Module):
    def __init__(self, d_in, d_out, hidden=(7, 7), dropout=0.1):
        super().__init__()
        layers, prev = [], d_in
        for i, h in enumerate(hidden):
            layers += [
                nn.Linear(prev, h),
                nn.ReLU(inplace=True),
            ]
            if i == len(hidden) - 1:
                layers.append(nn.BatchNorm1d(h))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, d_out))
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


def _train_one(model, Xt, Yt, Xv, Yv, cfg):
    opt = optim.SGD(model.parameters(),
                    lr=cfg["lr"], momentum=0.9, nesterov=True,
                    weight_decay=cfg["weight_decay"])
    crit = nn.MSELoss()

    bs = cfg.get("batch_size", 32)
    bs = max(2, min(bs, len(Xt) // 2))
    loader = DataLoader(TensorDataset(torch.tensor(Xt), torch.tensor(Yt)),
                        batch_size=bs, shuffle=True, drop_last=True)

    Xv_t = torch.tensor(Xv)
    Yv_t = torch.tensor(Yv)

    best_val, best_state, bad = float("inf"), None, 0
    for _ in range(cfg["epochs"]):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            vl = crit(model(Xv_t), Yv_t).item()
        if vl < best_val - 1e-8:
            best_val, bad = vl, 0
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


def train_ensemble(X_train, Y_train, cfg, hyper, n_seeds, top_k):
    X_train = np.asarray(X_train, dtype=np.float32)
    Y_train = np.asarray(Y_train, dtype=np.float32)
    n = len(X_train)
    n_tr = int(n * cfg["train_frac"])
    tr, va = np.arange(n_tr), np.arange(n_tr, n)

    sc_x = StandardScaler().fit(X_train[tr])
    Xt = sc_x.transform(X_train[tr]).astype(np.float32)
    Xv = sc_x.transform(X_train[va]).astype(np.float32)

    if STANDARDIZE_Y:
        y_mean = Y_train[tr].mean(axis=0)
        y_std  = Y_train[tr].std(axis=0) + 1e-8
        Yt = ((Y_train[tr] - y_mean) / y_std).astype(np.float32)
        Yv = ((Y_train[va] - y_mean) / y_std).astype(np.float32)
    else:
        y_mean = np.zeros(Y_train.shape[1], dtype=np.float64)
        y_std  = np.ones(Y_train.shape[1], dtype=np.float64)
        Yt = Y_train[tr].astype(np.float32)
        Yv = Y_train[va].astype(np.float32)

    d_in, d_out = Xt.shape[1], Yt.shape[1]
    models, losses = [], []
    for s in range(n_seeds):
        torch.manual_seed(s); np.random.seed(s)
        m = MLP(d_in, d_out, hidden=cfg["hidden"], dropout=hyper["dropout"])
        m, vl = _train_one(m, Xt, Yt, Xv, Yv,
                           {**cfg, "weight_decay": hyper["weight_decay"]})
        models.append(m); losses.append(vl)

    keep = np.argsort(losses)[:top_k]
    return [models[i] for i in keep], sc_x, y_mean, y_std, float(min(losses))


def predict_ensemble(models, X_new, sc_x, y_mean, y_std):
    X_new = sc_x.transform(np.asarray(X_new, dtype=np.float32)).astype(np.float32)
    x_t = torch.tensor(X_new)
    outs = []
    for m in models:
        m.eval()
        with torch.no_grad():
            outs.append(m(x_t).numpy())
    pred = np.mean(outs, axis=0)
    if STANDARDIZE_Y:
        pred = pred * y_std + y_mean
    return pred


def tune_hyperparameters(X_train, Y_train, cfg, grid):
    best_loss, best_hyper = np.inf, None
    for hyper in grid:
        try:
            _, _, _, _, vl = train_ensemble(
                X_train, Y_train, cfg, hyper,
                n_seeds=TUNE_SEEDS, top_k=2)
            if vl < best_loss:
                best_loss, best_hyper = vl, hyper
        except Exception:
            pass
    return best_hyper


def r2_oos(actual, forecast, benchmark):
    a, f, b = map(np.asarray, (actual, forecast, benchmark))
    ss_res = np.sum((a - f) ** 2)
    ss_tot = np.sum((a - b) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def newey_west_var(x, lags):
    x = np.asarray(x, dtype=float)
    n = len(x)
    e = x - x.mean()
    total = np.dot(e, e) / n
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        total += 2.0 * w * np.dot(e[lag:], e[:-lag]) / n
    return total / n


def clark_west(actual, forecast, benchmark, lags=CW_LAGS):
    a, f, b = map(np.asarray, (actual, forecast, benchmark))
    h = (a - b) ** 2 - (a - f) ** 2 + (b - f) ** 2
    var = newey_west_var(h, lags)
    if var <= 0:
        return np.nan, np.nan
    stat = h.mean() / np.sqrt(var)
    return float(stat), float(1.0 - stats.norm.cdf(stat))


def forecast_all(X, Y, cfg, first_origin, gap, hyper_grid, tune_every):
    common = X.index.intersection(Y.dropna().index)
    X, Y = X.loc[common], Y.loc[common]
    origins = X.index[X.index >= first_origin]
    mat_cols = list(Y.columns)
    n_mat = len(mat_cols)

    actual = np.full((len(origins), n_mat), np.nan)
    forecast = np.full_like(actual, np.nan)
    benchmark = np.full_like(actual, np.nan)

    current_hyper = dict(dropout=cfg["dropout"],
                         weight_decay=cfg["weight_decay"])
    last_tune_t = -tune_every
    n_tunings = 0
    t_start = time.time()

    for i, origin in enumerate(origins):
        hist_mask = X.index <= origin - gap
        X_hist = X[hist_mask].values
        Y_hist = Y[hist_mask].values
        X_now  = X.loc[[origin]].values
        if len(X_hist) < 50:
            continue

        if i - last_tune_t >= tune_every:
            t0 = time.time()
            best = tune_hyperparameters(X_hist, Y_hist, cfg, hyper_grid)
            if best is not None:
                current_hyper = best
            last_tune_t = i
            n_tunings += 1
            print(f"    [tune #{n_tunings}] t={origin} → "
                  f"dropout={current_hyper['dropout']}, "
                  f"wd={current_hyper['weight_decay']:.0e} "
                  f"({time.time()-t0:.0f}s)")

        ens, sc_x, y_mean, y_std, _ = train_ensemble(
            X_hist, Y_hist, cfg, current_hyper,
            n_seeds=cfg["n_seeds"], top_k=cfg["top_k"])

        pred = predict_ensemble(ens, X_now, sc_x, y_mean, y_std)
        actual[i, :] = Y.loc[origin].values
        forecast[i, :] = pred[0]
        benchmark[i, :] = Y_hist.mean(axis=0)

        if (i + 1) % 12 == 0:
            el = time.time() - t_start
            eta = el / (i + 1) * (len(origins) - i - 1)
            print(f"    [{i+1}/{len(origins)}] t={origin} "
                  f"elapsed={el:.0f}s ETA={eta:.0f}s")

    df = pd.DataFrame(index=origins)
    for j, c in enumerate(mat_cols):
        df[f"actual_{c}"]    = actual[:, j]
        df[f"forecast_{c}"]  = forecast[:, j]
        df[f"benchmark_{c}"] = benchmark[:, j]
    return df, mat_cols


def report(df, mat_cols):
    print()
    print("ИТОГ: R²_oos (%)")
    print(f"{'maturity':>10s}  {'R²':>8s}  {'p-value':>10s}  {'n_obs':>6s}")
    for c in mat_cols:
        a = df[f"actual_{c}"].values
        f = df[f"forecast_{c}"].values
        b = df[f"benchmark_{c}"].values
        mask = ~(np.isnan(a) | np.isnan(f) | np.isnan(b))
        r2 = r2_oos(a[mask], f[mask], b[mask])
        _, p = clark_west(a[mask], f[mask], b[mask])
        print(f"{c:>10s}  {100*r2:>+8.2f}  {p:>10.4f}  {mask.sum():>6d}")

    actuals    = np.column_stack([df[f"actual_{c}"].values    for c in mat_cols])
    forecasts  = np.column_stack([df[f"forecast_{c}"].values  for c in mat_cols])
    benchmarks = np.column_stack([df[f"benchmark_{c}"].values for c in mat_cols])
    mask = ~(np.isnan(actuals) | np.isnan(forecasts) | np.isnan(benchmarks))
    rows = mask.all(axis=1)
    r2_ew = r2_oos(actuals[rows].mean(1), forecasts[rows].mean(1),
                   benchmarks[rows].mean(1))
    _, p_ew = clark_west(actuals[rows].mean(1), forecasts[rows].mean(1),
                         benchmarks[rows].mean(1))

    print(f"{'EW':>10s}  {100*r2_ew:>+8.2f}  {p_ew:>10.4f}  {rows.sum():>6d}")

def main():
    print("# BBT (2020) NN — Table 1, yield-only, NN 2 Layer (7 nodes each)")
    print(f"# FIRST_ORIGIN={FIRST_ORIGIN}, GAP={GAP}")
    print(f"# STANDARDIZE_Y={STANDARDIZE_Y}  (по статье — сырые xr)")
    print(f"# seeds={NN_CFG['n_seeds']}, top_k={NN_CFG['top_k']}, "
          f"epochs={NN_CFG['epochs']}, patience={NN_CFG['patience']}")
    print(f"# hidden={NN_CFG['hidden']}, dropout={NN_CFG['dropout']}, "
          f"wd={NN_CFG['weight_decay']}, lr={NN_CFG['lr']}, "
          f"batch={NN_CFG['batch_size']}")
    print(f"# CV каждые {TUNE_EVERY} мес, grid={len(HYPER_GRID)} combos")

    t0 = time.time()
    X, Y = load_data()

    df, mat_cols = forecast_all(X, Y, NN_CFG, FIRST_ORIGIN, GAP,
                                HYPER_GRID, TUNE_EVERY)
    df.to_csv(RESULTS_DIR / "bbt_nn_table1_2layer7.csv")
    report(df, mat_cols)

    print()
    print(f"Общее время: {time.time()-t0:.0f}s "
          f"({(time.time()-t0)/60:.1f} мин)")
    print(f"Сохранено: {RESULTS_DIR}/bbt_nn_table1_2layer7.csv")


if __name__ == "__main__":
    main()