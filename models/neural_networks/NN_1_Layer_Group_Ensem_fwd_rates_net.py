"""
NN 1-Layer Group Ensem + Fwd. Rate Net
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

try:
    torch.set_num_threads(2)
except RuntimeError:
    pass

try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    ROOT = Path("/content")

LW_FILE = ROOT / "LW_monthly.xlsx"
MD_FILE = ROOT / "FRED-MD_2018m12.csv"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"ROOT = {ROOT}")
print(f"LW exists: {LW_FILE.exists()}")
print(f"MD exists: {MD_FILE.exists()}")

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
    n_nodes_per_group=1,
    fwd_hidden=3,
    dropout_group=0.1,
    dropout_fwd=0.0,
    weight_decay_group=1e-3,
    weight_decay_fwd=1e-4,
    lr=1e-2,
    n_seeds=50,
    top_k=10,
    epochs=500,
    patience=20,
    batch_size=32,
    train_frac=0.85,
)


HYPER_GRID = [
    dict(dropout_group=0.1, weight_decay_group=1e-2),
    dict(dropout_group=0.1, weight_decay_group=1e-3),
    dict(dropout_group=0.3, weight_decay_group=1e-2),
    dict(dropout_group=0.3, weight_decay_group=1e-3),
    dict(dropout_group=0.5, weight_decay_group=1e-2),
    dict(dropout_group=0.5, weight_decay_group=1e-3),
]
TUNE_EVERY = 60
TUNE_SEEDS = 5


FRED_MD_GROUPS = {
    "Output & Income": [
        "RPI", "W875RX1", "DPCERA3M086SBEA", "CMRMTSPLx", "RETAILx",
        "INDPRO", "IPFPNSS", "IPFINAL", "IPCONGD", "IPDCONGD", "IPNCONGD",
        "IPBUSEQ", "IPMAT", "IPDMAT", "IPNMAT", "IPMANSICS", "IPB51222S",
        "IPFUELS", "CUMFNS",
    ],
    "Labor Market": [
        "HWI", "HWIURATIO", "CLF16OV", "CE16OV", "UNRATE", "UEMPMEAN",
        "UEMPLT5", "UEMP5TO14", "UEMP15OV", "UEMP15T26", "UEMP27OV",
        "CLAIMSx", "PAYEMS", "USGOOD", "CES1021000001", "USCONS",
        "MANEMP", "DMANEMP", "NDMANEMP", "SRVPRD", "USTPU", "USWTRADE",
        "USTRADE", "USFIRE", "USGOVT", "CES0600000007", "AWOTMAN", "AWHMAN",
    ],
    "Housing": [
        "HOUST", "HOUSTNE", "HOUSTMW", "HOUSTS", "HOUSTW",
        "PERMIT", "PERMITNE", "PERMITMW", "PERMITS", "PERMITW",
    ],
    "Consumption, Orders & Inventories": [
        "ACOGNO", "AMDMNOx", "ANDENOx", "AMDMUOx", "BUSINVx", "ISRATIOx",
    ],
    "Money & Credit": [
        "M1SL", "M2SL", "M2REAL", "AMBSL", "TOTRESNS", "NONBORRES",
        "BUSLOANS", "REALLN", "NONREVSL", "CONSPI",
    ],
    "Stock Market": [
        "S&P 500", "S&P: indust", "S&P div yield", "S&P PE ratio",
    ],
    "Interest & Exchange Rates": [
        "FEDFUNDS", "CP3Mx", "TB3MS", "TB6MS", "GS1", "GS5", "GS10",
        "AAA", "BAA", "COMPAPFFx", "TB3SMFFM", "TB6SMFFM", "T1YFFM",
        "T5YFFM", "T10YFFM", "AAAFFM", "BAAFFM", "TWEXMMTH", "EXSZUSx",
        "EXJPUSx", "EXUSUKx", "EXCAUSx",
    ],
    "Prices": [
        "WPSFD49207", "WPSFD49502", "WPSID61", "WPSID62", "OILPRICEx",
        "PPICMM", "CPIAUCSL", "CPIAPPSL", "CPITRNSL", "CPIMEDSL",
        "CUSR0000SAC", "CUSR0000SAD", "CUSR0000SAS", "CPIULFSL",
        "CUSR0000SA0L2", "CUSR0000SA0L5", "PCEPI", "DDURRG3M086SBEA",
        "DNDGRG3M086SBEA", "DSERRG3M086SBEA", "CES0600000008",
        "CES2000000008", "CES3000000008", "UMCSENTx", "MZMSL",
        "DTCOLNVHFNM", "DTCTHFNM", "INVEST", "VXOCLSx",
    ],
}


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


def load_fred_md(path: Path) -> pd.DataFrame:
    print(f"[data] Читаю {path.name} (macro)...")
    raw = pd.read_csv(path, header=None, dtype=str)

    header = raw.iloc[0].tolist()
    header[0] = "date"
    tcodes_raw = raw.iloc[1].tolist()[1:]
    tcodes = {}
    for h, v in zip(header[1:], tcodes_raw):
        try:
            tcodes[h] = int(float(v))
        except Exception:
            tcodes[h] = 1

    data = raw.iloc[2:].copy()
    data.columns = header
    data = data.reset_index(drop=True)
    data["date"] = pd.PeriodIndex(pd.to_datetime(data["date"]), freq="M")
    data = data.set_index("date").sort_index()

    def _t(s, c):
        s = pd.to_numeric(s, errors="coerce")
        if c == 1:  return s
        if c == 2:  return s.diff()
        if c == 3:  return s.diff().diff()
        if c == 4:  return np.log(s.where(s > 1e-6))
        if c == 5:  return np.log(s.where(s > 1e-6)).diff()
        if c == 6:  return np.log(s.where(s > 1e-6)).diff().diff()
        if c == 7:  return s.pct_change(fill_method=None).diff()
        return s

    transformed = [_t(data[col], tcodes.get(col, 1)) for col in data.columns]
    out = pd.concat(transformed, axis=1)
    out.columns = data.columns
    out = out[out.index.notna()]

    print(f"  macro shape={out.shape}, {out.index[0]}..{out.index[-1]}")
    return out


def load_data():
    print("Данные")
    lw = load_liu_wu(LW_FILE)
    lo = pd.Period(SAMPLE_START, freq="M")
    hi = pd.Period(SAMPLE_END, freq="M")
    lw = lw[(lw.index >= lo) & (lw.index <= hi)]

    fwd = build_forward_rates(lw)
    exc = build_excess_returns(lw, MATURITIES, HOLDING)
    md  = load_fred_md(MD_FILE)

    common = fwd.index.intersection(exc.dropna().index).intersection(md.index)
    common = common[(common >= lo) & (common <= hi)]

    md_slice = md.loc[common]
    md_slice = md_slice.dropna(axis=1, how="any")
    dropped = md.shape[1] - md_slice.shape[1]
    print(f"  macro после dropna: {md_slice.shape[1]} колонок "
          f"(отброшено {dropped})")

    groups_data = []
    groups_names = []
    total_used = 0
    for gname, gcols in FRED_MD_GROUPS.items():
        present = [c for c in gcols if c in md_slice.columns]
        if not present:
            continue
        groups_data.append(md_slice[present])
        groups_names.append(gname)
        total_used += len(present)
        print(f"  Group '{gname}': {len(present)} cols")

    used_set = set()
    for g in groups_data:
        used_set.update(g.columns)
    missing = [c for c in md_slice.columns if c not in used_set]
    if missing:
        print(f"  [warn] {len(missing)} колонок вне групп: {missing[:5]}...")
        groups_data.append(md_slice[missing])
        groups_names.append("Misc")
        print(f"  Group 'Misc': {len(missing)} cols")

    X_fwd = fwd.loc[common]

    valid = np.ones(len(common), dtype=bool)
    for g in groups_data:
        valid &= ~g.isna().any(axis=1).values
    valid &= ~X_fwd.isna().any(axis=1).values
    Y = exc.loc[common]
    valid &= ~Y.isna().any(axis=1).values

    X_fwd = X_fwd[valid]
    Y = Y[valid]
    groups_data = [g[valid] for g in groups_data]

    print(f"\nX_fwd: {X_fwd.shape}")
    print(f"Y: {Y.shape}")
    print(f"Период: {X_fwd.index[0]}..{X_fwd.index[-1]}")
    print(f"Y stats: mean={Y.values.mean():.5f}, std={Y.values.std():.5f}")
    for c in Y.columns:
        print(f"  {c}: mean={Y[c].mean():+.5f}, std={Y[c].std():.5f}")

    return groups_data, groups_names, X_fwd, Y

class GroupEnsemMLP(nn.Module):
    """
      - для каждой группы макро: Linear(d_g, n_per_group) → ReLU → BatchNorm → Dropout
      - fwd net: Linear(d_fwd, fwd_hidden) → ReLU → BatchNorm → Dropout
      - конкатенация всех hidden → Linear(total, d_out)
    """
    def __init__(self, group_dims, d_fwd, d_out,
                 n_nodes_per_group=1, fwd_hidden=3,
                 dropout_group=0.1, dropout_fwd=0.0):
        super().__init__()

        self.group_nets = nn.ModuleList()
        for d_g in group_dims:
            layers = [
                nn.Linear(d_g, n_nodes_per_group),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(n_nodes_per_group),
            ]
            if dropout_group > 0:
                layers.append(nn.Dropout(dropout_group))
            self.group_nets.append(nn.Sequential(*layers))

        fwd_layers = [
            nn.Linear(d_fwd, fwd_hidden),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(fwd_hidden),
        ]
        if dropout_fwd > 0:
            fwd_layers.append(nn.Dropout(dropout_fwd))
        self.fwd_net = nn.Sequential(*fwd_layers)

        total_dim = n_nodes_per_group * len(group_dims) + fwd_hidden
        self.out = nn.Linear(total_dim, d_out)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_groups, x_fwd):
        h_parts = [net(xg) for net, xg in zip(self.group_nets, x_groups)]
        h_parts.append(self.fwd_net(x_fwd))
        h = torch.cat(h_parts, dim=1)
        return self.out(h)


def _train_one(model, Xg_t, Xf_t, Yt, Xg_v, Xf_v, Yv, cfg,
               dropout_group, dropout_fwd, wd_group, wd_fwd):
    params_group = list(model.group_nets.parameters()) + list(model.out.parameters())
    params_fwd   = list(model.fwd_net.parameters())
    opt = optim.SGD([
        {"params": params_group, "weight_decay": wd_group},
        {"params": params_fwd,   "weight_decay": wd_fwd},
    ], lr=cfg["lr"], momentum=0.9, nesterov=True)
    crit = nn.MSELoss()

    n = Xf_t.shape[0]
    bs = max(2, min(cfg["batch_size"], n // 2))
    n_batches = max(1, n // bs)

    best_val, best_state, bad = float("inf"), None, 0
    for _ in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(n)
        for b in range(n_batches):
            idx = perm[b * bs:(b + 1) * bs]
            xg_b = [xg[idx] for xg in Xg_t]
            xf_b = Xf_t[idx]
            y_b  = Yt[idx]
            opt.zero_grad()
            loss = crit(model(xg_b, xf_b), y_b)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            vl = crit(model(Xg_v, Xf_v), Yv).item()
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


def train_ensemble(Xg_tr, Xf_tr, Y_tr, cfg, hyper, n_seeds, top_k):
    n = len(Y_tr)
    n_tr = int(n * cfg["train_frac"])
    tr, va = np.arange(n_tr), np.arange(n_tr, n)

    sc_groups = [StandardScaler().fit(Xg[tr]) for Xg in Xg_tr]
    sc_fwd    = StandardScaler().fit(Xf_tr[tr])

    Xg_t_np = [sc.transform(Xg[tr]).astype(np.float32)
               for sc, Xg in zip(sc_groups, Xg_tr)]
    Xg_v_np = [sc.transform(Xg[va]).astype(np.float32)
               for sc, Xg in zip(sc_groups, Xg_tr)]
    Xf_t_np = sc_fwd.transform(Xf_tr[tr]).astype(np.float32)
    Xf_v_np = sc_fwd.transform(Xf_tr[va]).astype(np.float32)

    if STANDARDIZE_Y:
        y_mean = Y_tr[tr].mean(axis=0)
        y_std  = Y_tr[tr].std(axis=0) + 1e-8
        Yt_np = ((Y_tr[tr] - y_mean) / y_std).astype(np.float32)
        Yv_np = ((Y_tr[va] - y_mean) / y_std).astype(np.float32)
    else:
        y_mean = np.zeros(Y_tr.shape[1], dtype=np.float64)
        y_std  = np.ones(Y_tr.shape[1], dtype=np.float64)
        Yt_np = Y_tr[tr].astype(np.float32)
        Yv_np = Y_tr[va].astype(np.float32)

    Xg_t = [torch.from_numpy(Xg) for Xg in Xg_t_np]
    Xg_v = [torch.from_numpy(Xg) for Xg in Xg_v_np]
    Xf_t = torch.from_numpy(Xf_t_np)
    Xf_v = torch.from_numpy(Xf_v_np)
    Yt_t = torch.from_numpy(Yt_np)
    Yv_t = torch.from_numpy(Yv_np)

    group_dims = [Xg.shape[1] for Xg in Xg_t_np]
    d_fwd = Xf_t_np.shape[1]
    d_out = Yt_np.shape[1]

    models, losses = [], []
    for s in range(n_seeds):
        torch.manual_seed(s); np.random.seed(s)
        m = GroupEnsemMLP(
            group_dims, d_fwd, d_out,
            n_nodes_per_group=cfg["n_nodes_per_group"],
            fwd_hidden=cfg["fwd_hidden"],
            dropout_group=hyper["dropout_group"],
            dropout_fwd=cfg["dropout_fwd"],
        )
        m, vl = _train_one(m, Xg_t, Xf_t, Yt_t, Xg_v, Xf_v, Yv_t, cfg,
                           dropout_group=hyper["dropout_group"],
                           dropout_fwd=cfg["dropout_fwd"],
                           wd_group=hyper["weight_decay_group"],
                           wd_fwd=cfg["weight_decay_fwd"])
        models.append(m); losses.append(vl)

    keep = np.argsort(losses)[:top_k]
    return ([models[i] for i in keep],
            sc_groups, sc_fwd, y_mean, y_std, float(min(losses)))


def predict_ensemble(models, Xg_new, Xf_new, sc_groups, sc_fwd,
                     y_mean, y_std):
    Xg = [sc.transform(np.asarray(x, dtype=np.float32)).astype(np.float32)
          for sc, x in zip(sc_groups, Xg_new)]
    Xf = sc_fwd.transform(np.asarray(Xf_new, dtype=np.float32)).astype(np.float32)
    xg_t = [torch.from_numpy(x) for x in Xg]
    xf_t = torch.from_numpy(Xf)
    outs = []
    for m in models:
        m.eval()
        with torch.no_grad():
            outs.append(m(xg_t, xf_t).numpy())
    pred = np.mean(outs, axis=0)
    if STANDARDIZE_Y:
        pred = pred * y_std + y_mean
    return pred


def tune_hyperparameters(Xg, Xf, Y, cfg, grid):
    best_loss, best_hyper = np.inf, None
    for hyper in grid:
        try:
            _, _, _, _, _, vl = train_ensemble(
                Xg, Xf, Y, cfg, hyper,
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


def forecast_all(groups_data, X_fwd, Y, cfg, first_origin, gap,
                 hyper_grid, tune_every):
    common = X_fwd.index.intersection(Y.index)
    for g in groups_data:
        common = common.intersection(g.index)

    Xg_all = [g.loc[common].values for g in groups_data]
    Xf_all = X_fwd.loc[common].values
    Y_all  = Y.loc[common]

    origins = X_fwd.loc[common].index
    origins = origins[origins >= first_origin]
    mat_cols = list(Y_all.columns)
    n_mat = len(mat_cols)
    idx_pos = {o: i for i, o in enumerate(X_fwd.loc[common].index)}

    actual    = np.full((len(origins), n_mat), np.nan)
    forecast  = np.full_like(actual, np.nan)
    benchmark = np.full_like(actual, np.nan)

    current_hyper = dict(dropout_group=cfg["dropout_group"],
                         weight_decay_group=cfg["weight_decay_group"])
    last_tune_t = -tune_every
    n_tunings = 0
    t_start = time.time()

    for i, origin in enumerate(origins):
        p = idx_pos[origin]
        hist_end = p - gap + 1
        if hist_end < 50:
            continue

        Xg_hist = [Xg[:hist_end] for Xg in Xg_all]
        Xf_hist = Xf_all[:hist_end]
        Y_hist  = Y_all.iloc[:hist_end].values
        Xg_now  = [Xg[p:p + 1] for Xg in Xg_all]
        Xf_now  = Xf_all[p:p + 1]

        if i - last_tune_t >= tune_every:
            t0 = time.time()
            best = tune_hyperparameters(Xg_hist, Xf_hist, Y_hist,
                                        cfg, hyper_grid)
            if best is not None:
                current_hyper = best
            last_tune_t = i
            n_tunings += 1
            print(f"    [tune #{n_tunings}] t={origin} → "
                  f"dropout={current_hyper['dropout_group']}, "
                  f"wd={current_hyper['weight_decay_group']:.0e} "
                  f"({time.time()-t0:.0f}s)", flush=True)

        ens, sc_g, sc_f, y_mean, y_std, _ = train_ensemble(
            Xg_hist, Xf_hist, Y_hist, cfg, current_hyper,
            n_seeds=cfg["n_seeds"], top_k=cfg["top_k"])

        pred = predict_ensemble(ens, Xg_now, Xf_now, sc_g, sc_f,
                                y_mean, y_std)
        actual[i, :] = Y_all.iloc[p].values
        forecast[i, :] = pred[0]
        benchmark[i, :] = Y_hist.mean(axis=0)

        if (i + 1) % 6 == 0:
            el = time.time() - t_start
            eta = el / (i + 1) * (len(origins) - i - 1)
            print(f"    [{i+1}/{len(origins)}] t={origin} "
                  f"elapsed={el:.0f}s ETA={eta:.0f}s", flush=True)

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
    print("# BBT (2020) — Table 2, NN 1 Layer Group Ensem + fwd rate net")
    print(f"# n_nodes_per_group={NN_CFG['n_nodes_per_group']}, "
          f"fwd_hidden={NN_CFG['fwd_hidden']}")
    print(f"# FIRST_ORIGIN={FIRST_ORIGIN}, GAP={GAP}")
    print(f"# seeds={NN_CFG['n_seeds']}, top_k={NN_CFG['top_k']}, "
          f"epochs={NN_CFG['epochs']}, patience={NN_CFG['patience']}")
    print(f"# dropout_fwd={NN_CFG['dropout_fwd']}, "
          f"wd_fwd={NN_CFG['weight_decay_fwd']}")
    print(f"# CV каждые {TUNE_EVERY} мес, grid={len(HYPER_GRID)} combos")

    t0 = time.time()
    groups_data, groups_names, X_fwd, Y = load_data()

    print(f"\nВсего групп: {len(groups_data)}")
    print(f"Имена: {groups_names}")

    df, mat_cols = forecast_all(groups_data, X_fwd, Y, NN_CFG,
                                FIRST_ORIGIN, GAP, HYPER_GRID, TUNE_EVERY)
    df.to_csv(RESULTS_DIR / "bbt_nn_table2_group_ensem.csv")
    report(df, mat_cols)

    print()
    print(f"Общее время: {time.time()-t0:.0f}s "
          f"({(time.time()-t0)/60:.1f} мин)")
    print(f"Сохранено: {RESULTS_DIR}/bbt_nn_table2_group_ensem.csv")


if __name__ == "__main__":
    main()