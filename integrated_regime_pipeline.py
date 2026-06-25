# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Integrated Regime Modelling Pipeline
#
# This notebook-style script combines:
#
# 1. The root daily factor and GMM regime workflow from `regime_analysis.ipynb`.
# 2. The monthly metrics-state regime workflow from `regime_metrics_states.ipynb`.
# 3. The regime-aware CVaR allocation workflow from `CVAR_DR_RS_RSDR_Evaluation_paper_gamma_grid.ipynb`.
#
# Shared helper modules now live under `src/`, so the integrated script can
# read the consolidated code path directly instead of reaching into the legacy
# workflow folders at runtime. The file stays in Jupytext percent format so it
# can be exported cleanly to `.ipynb`.

# %%
from __future__ import annotations

import hashlib
import os
import sys
import time
import warnings
from pathlib import Path
from collections import defaultdict

for _v in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_v] = "1"
os.environ["PYTHONHASHSEED"] = "0"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from scipy.optimize import linprog
from scipy.special import logsumexp
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.inspection import permutation_importance
from sklearn.metrics import adjusted_rand_score, f1_score, precision_score, recall_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    ROOT = Path.cwd().resolve()

ROOT = ROOT.resolve()
DATA_ROOT = ROOT / "Data"
OUT_ROOT = ROOT / "outputs" / "integrated"
ROOT_GMM_OUT = OUT_ROOT / "root_gmm"
ROOT_GMM_FIG = ROOT_GMM_OUT / "figures"
REGIME_OUT = OUT_ROOT / "regime_states"
CVAR_OUT = OUT_ROOT / "cvar"

for _p in (ROOT_GMM_OUT, ROOT_GMM_FIG, REGIME_OUT, CVAR_OUT):
    _p.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))

from src import data_cleaner as dc
from src import data_loader as dl
from src import factor_construction as fc
from src import regime_model as rm
from src import regime_utils as ru
from src import regime_taa as rt

START_DATE = "2002-01-02"
INIT_END = pd.Timestamp("2017-12-01")
WIN_1976 = pd.Timestamp("1976-01-01")
ETA = 0.95
TRAIN_WINDOW = 120
REBALANCE_STEP = 1
GAMMA_GRID = [0.02, 0.04, 0.06, 0.08, 0.10]
SELECTED_REGIME_SOURCE = "altdata_bnmeanrank_hmm"
USE_WALKFORWARD_STATES = True

FEATURE_DESC = {
    "IR_dtb3": "3m Treasury yield change",
    "IR_dtb6": "6m Treasury yield change",
    "IR_dgs1": "1y Treasury yield change",
    "IR_dgs2": "2y Treasury yield change",
    "FF_RF": "Fama-French risk-free rate",
    "FF_Mkt_RF": "Fama-French market excess return",
    "FF_RMW": "Fama-French profitability factor",
    "BAB_USA": "Betting-against-beta factor",
    "BLM_SPGSCI": "S&P GSCI commodity return",
}

FEATURE_SHORT = {k: v for k, v in FEATURE_DESC.items()}
PC_SHORT = {
    "PC1": "rate momentum",
    "PC2": "quality / low-beta",
    "PC3": "reflation / risk-on",
    "PC4": "rate level",
    "PC5": "equity vs commodity spread",
    "PC6": "defensive style spread",
    "PC7": "front-end twist",
}


def ensure_required_inputs() -> dict[str, Path]:
    files = {
        "fredmd": DATA_ROOT / "fredmd_current.csv",
        "nber": DATA_ROOT / "nber_usrec.csv",
        "altdata_monthly": DATA_ROOT / "altdata_monthly.csv",
        "altdata_tcodes": DATA_ROOT / "altdata_tcodes.csv",
        "etf_returns": DATA_ROOT / "etf_returns.csv",
    }
    missing = [k for k, p in files.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")
    return files


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inventory_table(files: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for name, path in files.items():
        rows.append(
            {
                "name": name,
                "path": str(path.relative_to(ROOT)),
                "exists": path.exists(),
                "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
                "sha256": hash_file(path)[:12],
            }
        )
    return pd.DataFrame(rows)


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = df.to_frame() if isinstance(df, pd.Series) else df
    if path.suffix.lower() == ".parquet":
        obj.to_parquet(path)
    else:
        obj.to_csv(path)


print("Project root:", ROOT)
FILES = ensure_required_inputs()
display(inventory_table(FILES))

# %% [markdown]
# ## 1. Root Daily GMM
#
# This section consolidates the daily workflow from `regime_analysis.ipynb`:
# it cleans the root data panel, builds the factor matrix, fits the regime
# model, and writes the daily regime artefacts into `outputs/integrated/root_gmm/`.

# %%
print("=== Daily factor pipeline ===", flush=True)
dataset = dc.build_clean_dataset(start=START_DATE)
prices = dataset["prices"]
returns = dataset["returns"]
ff5 = dataset["ff5"]
rf = dataset["rf"]
calendar = dataset["calendar"]

factor_matrix = fc.build_factor_matrix(start=START_DATE, verbose=True)
X = fc.get_factor_matrix_for_gmm(factor_matrix, dropna=True)
model = rm.fit_regime_model(factor_matrix=X, n_components=None, run_cv=True, verbose=True)

ann_mean = factor_matrix.mean() * 252 * 100
ann_vol = factor_matrix.std() * (252**0.5) * 100
sharpe = ann_mean / ann_vol
summary = pd.DataFrame({"Mean_%": ann_mean, "Vol_%": ann_vol, "Sharpe": sharpe})
summary.to_csv(ROOT_GMM_OUT / "factor_summary_stats.csv")

corr = factor_matrix.dropna().corr()
corr.to_csv(ROOT_GMM_OUT / "factor_correlation.csv")

fig, axes = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
checks = [
    ("MXCXDMHR", "Equity (MSCI ACWI Hedged)"),
    ("LGY7TRUH", "Interest Rates (Global Govt 7-10yr)"),
    ("LUACTRUU", "Credit IG (US Corporate)"),
    ("BCOMTR", "Commodities (BCOM TR)"),
]
for ax, (col, label) in zip(axes, checks):
    if col in returns.columns:
        cum = (1 + returns[col]).cumprod()
        ax.plot(cum.index, cum.values, linewidth=0.8)
        ax.set_ylabel(label, fontsize=8)
        ax.grid(alpha=0.3)
axes[-1].set_xlabel("Date")
fig.suptitle("Cumulative Returns - Raw Series", fontsize=12)
plt.tight_layout()
fig.savefig(ROOT_GMM_FIG / "01_raw_cumulative_returns.png", dpi=120, bbox_inches="tight")
plt.close(fig)

regime_labels = model.hard_labels
regime_names = model.regime_names
regime_probs = model.probabilities
current_regime = rm.predict_current_regime(model, X, window_days=60)
means_df, vols_df = ru.compute_regime_stats(X, regime_labels, regime_names)
trans_df = ru.compute_transition_matrix(regime_labels, regime_names)
periods_df = ru.get_regime_periods(regime_labels, regime_names)

ROOT_GMM_OUT.mkdir(parents=True, exist_ok=True)
save_dataframe(prices, ROOT_GMM_OUT / "prices.parquet")
save_dataframe(returns, ROOT_GMM_OUT / "returns.parquet")
save_dataframe(factor_matrix, ROOT_GMM_OUT / "factor_matrix.parquet")
save_dataframe(X, ROOT_GMM_OUT / "gmm_input_X.parquet")
save_dataframe(regime_probs, ROOT_GMM_OUT / "regime_probabilities.parquet")
save_dataframe(regime_labels.rename("Regime"), ROOT_GMM_OUT / "regime_hard_labels.parquet")
regime_names_df = pd.DataFrame({"regime": regime_labels.map(regime_names)})
save_dataframe(regime_names_df, ROOT_GMM_OUT / "regime_named_labels.parquet")
regime_names_df.to_csv(ROOT_GMM_OUT / "regime_named_labels.csv")
trans_df.to_csv(ROOT_GMM_OUT / "regime_transition_matrix.csv")
current_regime.to_csv(ROOT_GMM_OUT / "current_regime.csv")
means_df.to_csv(ROOT_GMM_OUT / "regime_factor_means.csv")
vols_df.to_csv(ROOT_GMM_OUT / "regime_factor_vols.csv")
periods_df.to_csv(ROOT_GMM_OUT / "regime_periods.csv", index=False)
ff5.to_csv(ROOT_GMM_OUT / "ff5_daily.csv")
rf.to_csv(ROOT_GMM_OUT / "risk_free_daily.csv")

freq = regime_labels.value_counts().sort_index()
freq_df = pd.DataFrame(
    [
        {
            "regime": regime_names.get(k, f"Regime_{k}"),
            "days": int(cnt),
            "pct": round(100 * cnt / len(regime_labels), 2),
        }
        for k, cnt in freq.items()
    ]
)
freq_df.to_csv(ROOT_GMM_OUT / "regime_frequency.csv", index=False)

fig = ru.plot_dashboard(model, X, figsize=(18, 12))
fig.savefig(ROOT_GMM_FIG / "07_dashboard.png", dpi=140, bbox_inches="tight")
plt.close(fig)

fig = ru.plot_factor_heatmap(means_df, vols_df, title="Annualised Factor Mean Returns by Regime (%)")
fig.savefig(ROOT_GMM_FIG / "04_factor_heatmap.png", dpi=140, bbox_inches="tight")
plt.close(fig)

fig = ru.plot_regime_timeline(regime_labels, regime_names, title="Regime Classification History", figsize=(16, 2.5))
fig.savefig(ROOT_GMM_FIG / "05_regime_timeline.png", dpi=150, bbox_inches="tight")
plt.close(fig)

fig = ru.plot_regime_probabilities(regime_probs, regime_names, title="Regime Probabilities Over Time", figsize=(16, 4))
fig.savefig(ROOT_GMM_FIG / "06_regime_probabilities.png", dpi=150, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(13, 11))
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
ax.set_xticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(corr.index)))
ax.set_yticklabels(corr.index, fontsize=8)
ax.set_title("Factor Correlation Matrix (target: near-zero off-diagonal)", fontsize=11)
for i in range(len(corr)):
    for j in range(len(corr.columns)):
        ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=6)
plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
fig.savefig(ROOT_GMM_FIG / "02_factor_correlation.png", dpi=120, bbox_inches="tight")
plt.close(fig)

print("Current regime probabilities:")
display(current_regime)

# %% [markdown]
# ## 2. Monthly Metrics-State Regime Model
#
# This stage mirrors the JL monthly notebook. It builds the five feature
# bundles, evaluates GMM and HMM crisis flags against NBER recessions using
# Recall, Precision, and F1, and writes the final six-state HMM artefacts to
# `outputs/integrated/regime_states/`.

# %%
print("=== Monthly regime pipeline ===", flush=True)
data, tcodes = rt.load_fredmd(str(FILES["fredmd"]))
usrec = rt.load_usrec(str(FILES["nber"]))
alt_all = pd.read_csv(FILES["altdata_monthly"], index_col=0, parse_dates=True)
alt_tc_raw = pd.read_csv(FILES["altdata_tcodes"])
alt_col = alt_tc_raw.columns[0]
alt_tc = dict(zip(alt_tc_raw[alt_col], alt_tc_raw["tcode"]))

ALT_COLS = [c for c in alt_all.columns if not c.startswith("FRX_")]
MONTH_IDX = data.loc[WIN_1976:].index


def _g(col: str) -> pd.Series:
    if col not in data.columns:
        raise KeyError(f"Missing FRED-MD column: {col}")
    return data[col]


def engineered_features() -> dict[str, tuple[pd.Series, int]]:
    cpi_yoy = 100 * _g("CPIAUCSL").pct_change(12)
    u3 = _g("UNRATE").rolling(3).mean()
    ey = 100.0 / _g("S&P PE ratio")
    return {
        "YC_10Y3M": (_g("GS10") - _g("TB3MS"), 1),
        "YC_10Y1Y": (_g("GS10") - _g("GS1"), 1),
        "YC_5Y3M": (_g("GS5") - _g("TB3MS"), 1),
        "YC_10Y_FF": (_g("GS10") - _g("FEDFUNDS"), 1),
        "CREDIT_BAA10Y": (_g("BAA") - _g("GS10"), 1),
        "CREDIT_BAA_AAA": (_g("BAA") - _g("AAA"), 1),
        "CREDIT_AAA10Y": (_g("AAA") - _g("GS10"), 1),
        "REAL_FF": (_g("FEDFUNDS") - cpi_yoy, 1),
        "REAL_10Y": (_g("GS10") - cpi_yoy, 1),
        "SAHM": (u3 - u3.rolling(12).min(), 1),
        "UNRATE_12M_CHG": (_g("UNRATE") - _g("UNRATE").shift(12), 1),
        "SPX_MOM12": (100 * _g("S&P 500").pct_change(12), 1),
        "SPX_TREND": (100 * (_g("S&P 500") / _g("S&P 500").rolling(12).mean() - 1), 1),
        "SPX_PE": (_g("S&P PE ratio"), 1),
        "SPX_DY": (_g("S&P div yield"), 1),
        "ERP": (ey - (_g("GS10") - cpi_yoy), 1),
        "INFL_YoY": (cpi_yoy, 1),
        "OIL_MOM12": (100 * _g("OILPRICEx").pct_change(12), 1),
        "M2_REAL_YoY": (100 * (_g("M2SL") / _g("CPIAUCSL")).pct_change(12), 1),
        "CREDIT_GROWTH": (100 * _g("BUSLOANS").pct_change(12), 1),
        "VIX": (_g("VIXCLSx"), 4),
    }


ENG = {k: (s.dropna(), tc) for k, (s, tc) in engineered_features().items()}


def build_bundle(include_macro: bool, include_eng: bool, include_alt: bool) -> rt.FeatureBundle:
    idx = MONTH_IDX
    d = pd.DataFrame(index=idx)
    tc = pd.Series(dtype=float)
    if include_macro:
        for c in data.columns:
            d[c] = data.loc[idx, c]
            tc[c] = tcodes.get(c, 1)
    if include_eng:
        for k, (s, t) in ENG.items():
            d[k] = s.reindex(idx)
            tc[k] = t
    if include_alt:
        a = alt_all[ALT_COLS].reindex(idx)
        for c in a.columns:
            if a[c].isna().mean() <= 0.50:
                d[c] = a[c]
                tc[c] = alt_tc.get(c, 1)
    return rt.prepare_features(d, tc, exclude="exchange", pca_var=0.95, drop_initial=2)


fb_macro = build_bundle(True, False, False)
fb_macroeng = build_bundle(True, True, False)
fb_comb = build_bundle(True, True, True)
fb_alt = build_bundle(False, False, True)

print("Bundle sizes:")
for _name, _fb in [
    ("macro", fb_macro),
    ("macro+eng", fb_macroeng),
    ("comb", fb_comb),
    ("alt", fb_alt),
]:
    print(_name, _fb.transformed.shape, "->", _fb.scores.shape)


def regime_labels_until(scores_df: pd.DataFrame, end: pd.Timestamp) -> pd.Series:
    sc = scores_df.loc[:end]
    mdl = rt.RegimeModel(r=5, random_state=0, outlier_method="quantile", outlier_frac=0.15).fit(sc.values)
    return pd.Series(mdl.labels_, index=sc.index)


def rank_features(fb: rt.FeatureBundle, labels: pd.Series) -> pd.Series:
    Xdf = pd.DataFrame(
        fb.scaler.transform(fb.transformed.values),
        index=fb.transformed.index,
        columns=fb.columns,
    )
    common = Xdf.index.intersection(labels.index)
    X = Xdf.loc[common].values
    y = labels.loc[common].values
    cols = list(fb.columns)
    ranks = []
    for seed in range(3):
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_features="sqrt",
            min_samples_leaf=3,
            random_state=seed,
            n_jobs=1,
        ).fit(X, y)
        gini = pd.Series(rf_model.feature_importances_, index=cols)
        perm = pd.Series(
            permutation_importance(rf_model, X, y, n_repeats=3, random_state=seed, n_jobs=1).importances_mean,
            index=cols,
        )
        fstat = pd.Series(np.nan_to_num(f_classif(X, y)[0]), index=cols)
        ranks.append((gini.rank(ascending=False) + perm.rank(ascending=False) + fstat.rank(ascending=False)) / 3)
    return pd.concat(ranks, axis=1).mean(axis=1).sort_values()


def screen_tstats(fb: rt.FeatureBundle, end: pd.Timestamp) -> pd.Series:
    out = {}
    for col in fb.columns:
        feat = fb.transformed[col].dropna().loc[:end]
        idx = feat.index.intersection(usrec.index)
        if len(idx) < 10:
            out[col] = 0.0
            continue
        x = feat.loc[idx].values.astype(float)
        y = (usrec.loc[idx] > 0).astype(float).values
        if x.std() == 0 or y.std() == 0:
            out[col] = 0.0
            continue
        r = np.corrcoef(x, y)[0, 1]
        n = len(idx)
        out[col] = float(abs(r) * np.sqrt((n - 2) / max(1e-12, 1 - r**2)))
    return pd.Series(out).sort_values(ascending=False)


def bn_select(fb: rt.FeatureBundle, label_fb: rt.FeatureBundle) -> list[str]:
    ts = screen_tstats(fb, INIT_END)
    k = int((ts > np.sqrt(2 * np.log(len(fb.columns)))).sum())
    k = max(1, k)
    order = rank_features(fb, regime_labels_until(label_fb.scores, INIT_END)).index
    return [c for c in order if c in fb.transformed.columns][:k]


def subset_pca(fb: rt.FeatureBundle, cols: list[str]) -> tuple[pd.DataFrame, list[str], PCA]:
    avail = [c for c in cols if c in fb.transformed.columns]
    z = StandardScaler().fit_transform(fb.transformed[avail].values)
    full = PCA(svd_solver="full").fit(z)
    cum = np.cumsum(full.explained_variance_ratio_)
    n = int(np.searchsorted(cum, 0.95) + 1)
    pca = PCA(n_components=n, svd_solver="full").fit(z)
    scores = pd.DataFrame(pca.transform(z), index=fb.transformed.index, columns=[f"PC{i+1}" for i in range(n)])
    return scores, avail, pca


def scores_for(fb: rt.FeatureBundle, sel: str, label_fb: rt.FeatureBundle | None = None):
    if sel == "all":
        return fb.scores, list(fb.transformed.columns), None
    cols = bn_select(fb, label_fb or fb)
    return subset_pca(fb, cols)


def hmm_log_emissions(X: np.ndarray, means: np.ndarray, log_vars: np.ndarray) -> np.ndarray:
    T, d = X.shape
    K = means.shape[0]
    out = np.zeros((T, K))
    for k in range(K):
        diff = X - means[k]
        out[:, k] = -0.5 * (
            np.sum(diff**2 * np.exp(-log_vars[k]), axis=1) + np.sum(log_vars[k]) + d * np.log(2 * np.pi)
        )
    return out


def hmm_forward(log_b: np.ndarray, log_A: np.ndarray, log_pi: np.ndarray) -> np.ndarray:
    T, K = log_b.shape
    la = np.zeros((T, K))
    la[0] = log_pi + log_b[0]
    for t in range(1, T):
        la[t] = logsumexp(la[t - 1, :, None] + log_A, axis=0) + log_b[t]
    return la


def hmm_backward(log_b: np.ndarray, log_A: np.ndarray) -> np.ndarray:
    T, K = log_b.shape
    lb = np.zeros((T, K))
    for t in range(T - 2, -1, -1):
        lb[t] = logsumexp(log_A + log_b[t + 1] + lb[t + 1], axis=1)
    return lb


def fit_hmm(X: np.ndarray, n_states: int = 6, n_iter: int = 20, n_init: int = 2, reg: float = 1e-2, seed: int = 0):
    T, d = X.shape
    K = n_states
    rng = np.random.default_rng(seed)
    best_ll = -np.inf
    best = None
    for _ in range(n_init):
        km = KMeans(K, n_init=3, random_state=int(rng.integers(10000))).fit(X)
        means = km.cluster_centers_.copy()
        log_vars = np.log(np.full((K, d), X.var(axis=0).clip(1e-6)) + reg)
        A = np.full((K, K), 0.05 / max(1, K - 1))
        np.fill_diagonal(A, 0.95)
        pi = np.ones(K) / K
        prev = -np.inf
        for _ in range(n_iter):
            log_b = hmm_log_emissions(X, means, log_vars)
            la = hmm_forward(log_b, np.log(A + 1e-300), np.log(pi + 1e-300))
            lb = hmm_backward(log_b, np.log(A + 1e-300))
            ll = float(logsumexp(la[-1]))
            lg = la + lb
            lg -= logsumexp(lg, axis=1, keepdims=True)
            gamma = np.exp(lg)
            xi = np.zeros((K, K))
            for t in range(T - 1):
                log_xi = la[t, :, None] + np.log(A + 1e-300) + log_b[t + 1] + lb[t + 1]
                xi += np.exp(log_xi - logsumexp(log_xi))
            gamma_sum = gamma.sum(axis=0).clip(1e-10)
            pi = gamma[0] / gamma[0].sum()
            A = xi / xi.sum(axis=1, keepdims=True).clip(1e-10)
            means = (gamma[:, :, None] * X[:, None, :]).sum(axis=0) / gamma_sum[:, None]
            for k in range(K):
                diff = X - means[k]
                var = (gamma[:, k, None] * diff**2).sum(axis=0) / gamma_sum[k] + reg
                log_vars[k] = np.log(var)
            if abs(ll - prev) < 1e-3:
                break
            prev = ll
        if ll > best_ll:
            best_ll = ll
            best = (means.copy(), log_vars.copy(), A.copy(), pi.copy())
    return (*best, best_ll)


def hmm_filter(X: np.ndarray, means: np.ndarray, log_vars: np.ndarray, A: np.ndarray, pi: np.ndarray) -> np.ndarray:
    la = hmm_forward(hmm_log_emissions(X, means, log_vars), np.log(A + 1e-300), np.log(pi + 1e-300))
    la -= logsumexp(la, axis=1, keepdims=True)
    return np.exp(la)


def hmm_viterbi(X: np.ndarray, means: np.ndarray, log_vars: np.ndarray, A: np.ndarray, pi: np.ndarray) -> np.ndarray:
    T = X.shape[0]
    K = means.shape[0]
    log_b = hmm_log_emissions(X, means, log_vars)
    log_A = np.log(A + 1e-300)
    delta = np.zeros((T, K))
    psi = np.zeros((T, K), dtype=int)
    delta[0] = np.log(pi + 1e-300) + log_b[0]
    for t in range(1, T):
        sc = delta[t - 1, :, None] + log_A
        psi[t] = sc.argmax(axis=0)
        delta[t] = sc.max(axis=0) + log_b[t]
    states = np.zeros(T, dtype=int)
    states[-1] = int(delta[-1].argmax())
    for t in range(T - 2, -1, -1):
        states[t] = psi[t + 1, states[t + 1]]
    return states


def ownership_crisis(labels_tr: np.ndarray, Xtr: np.ndarray, frac: float = 0.15) -> int:
    mu = Xtr.mean(axis=0)
    VI = np.linalg.pinv(np.cov(Xtr.T) + 1e-6 * np.eye(Xtr.shape[1]))
    df = Xtr - mu
    dist = np.einsum("ij,jk,ik->i", df, VI, df)
    ext = dist >= np.quantile(dist, 1 - frac)
    k = int(labels_tr.max()) + 1
    return int(np.argmax([int(ext[labels_tr == i].sum()) for i in range(k)]))


def binary_regime_metrics(flag: pd.Series, idx: pd.Index) -> tuple[float, float, float]:
    y_true = (usrec.reindex(idx).fillna(0) > 0).astype(int).values
    y_pred = pd.Series(flag, index=idx).astype(int).values
    return (
        float(recall_score(y_true, y_pred, zero_division=0)),
        float(precision_score(y_true, y_pred, zero_division=0)),
        float(f1_score(y_true, y_pred, zero_division=0)),
    )


def bicscore(ll: float, n_obs: int, n_states: int, n_features: int) -> float:
    params = (n_states - 1) + n_states * (n_states - 1) + 2 * n_states * n_features
    return float(-2 * ll + params * np.log(max(1, n_obs)))


def fit_engine(sc: pd.DataFrame, engine: str):
    X = sc.values.astype(float)
    if engine.upper() == "GMM":
        gm = GaussianMixture(
            n_components=6,
            covariance_type="diag",
            n_init=10,
            random_state=0,
            reg_covar=1e-6,
            max_iter=200,
        ).fit(X)
        states = gm.predict(X)
        probs = gm.predict_proba(X)
        crisis = ownership_crisis(states, X)
        return pd.Series(states, index=sc.index), pd.DataFrame(probs, index=sc.index), crisis, gm, float(gm.bic(X))
    means, log_vars, A, pi, ll = fit_hmm(X, n_states=6, seed=0)
    states = hmm_viterbi(X, means, log_vars, A, pi)
    probs = hmm_filter(X, means, log_vars, A, pi)
    crisis = ownership_crisis(states, X)
    return pd.Series(states, index=sc.index), pd.DataFrame(probs, index=sc.index), crisis, (means, log_vars, A, pi, ll), bicscore(ll, len(X), 6, X.shape[1])


SPECS = [
    ("macro+eng", fb_macroeng, "all", None),
    ("comb_all/all", fb_comb, "all", None),
    ("comb_all/bn_meanrank", fb_comb, "bn_meanrank", fb_macro),
    ("altdata_full/all", fb_alt, "all", None),
    ("altdata_full/bn_meanrank", fb_alt, "bn_meanrank", fb_macro),
]

metric_rows = []
SC = {}
FITS = {}
path_fig, path_axes = plt.subplots(len(SPECS), 2, figsize=(13, 12), sharex=True)

print("=== Metrics-state comparison across five configs x GMM/HMM ===", flush=True)
for i, (name, fb, sel, label_fb) in enumerate(SPECS):
    sc, feats_used, _ = scores_for(fb, sel, label_fb=label_fb)
    sc = sc.dropna()
    SC[name] = (sc, feats_used)
    n_rec = int((usrec.reindex(sc.index).fillna(0) > 0).sum())
    engine_scores = {}
    for engine in ("GMM", "HMM"):
        states, probs, crisis, fitobj, bic = fit_engine(sc, engine)
        FITS[(name, engine)] = (states, crisis, fitobj, bic)
        crisis_flag = (states == crisis).astype(int)
        rec, prec, f1 = binary_regime_metrics(crisis_flag, sc.index)
        engine_scores[engine] = {"rec": rec, "prec": prec, "f1": f1}
        ax = path_axes[i, 0 if engine == "GMM" else 1]
        ax.plot(states.index, states.values, lw=0.8)
        ax.set_title(f"{name} - {engine}  rec={rec:.3f}  prec={prec:.3f}  f1={f1:.3f}")
        ax.set_yticks(sorted(states.unique()))
        ax.grid(alpha=0.2)
        if i == len(SPECS) - 1:
            ax.tick_params(axis="x", rotation=45)
    metric_rows.append(
        {
            "config": name,
            "n_feat": len(feats_used),
            "n_PCs": sc.shape[1],
            "n_rec": n_rec,
            "Recall_GMM": engine_scores["GMM"]["rec"],
            "Recall_HMM": engine_scores["HMM"]["rec"],
            "Precision_GMM": engine_scores["GMM"]["prec"],
            "Precision_HMM": engine_scores["HMM"]["prec"],
            "F1_GMM": engine_scores["GMM"]["f1"],
            "F1_HMM": engine_scores["HMM"]["f1"],
        }
    )

path_fig.tight_layout()
path_fig.savefig(REGIME_OUT / "fig_state_paths.png", dpi=140, bbox_inches="tight")
plt.close(path_fig)

metrics_df = pd.DataFrame(metric_rows)
metrics_df.to_csv(REGIME_OUT / "regime_metrics_table.csv", index=False)
metrics_df.to_csv(REGIME_OUT / "regime_kappa_table.csv", index=False)

diagnostic_rows = []
for eng in ("GMM", "HMM"):
    for name, fb, sel, label_fb in SPECS:
        sc, feats_used = SC[name]
        states, crisis, fitobj, bic = FITS[(name, eng)]
        ari_scores = []
        for seed in range(5):
            if eng == "GMM":
                gm = GaussianMixture(
                    n_components=6,
                    covariance_type="diag",
                    n_init=5,
                    random_state=seed,
                    reg_covar=1e-6,
                    max_iter=200,
                ).fit(sc.values)
                ari_scores.append(adjusted_rand_score(states, gm.predict(sc.values)))
            else:
                means, log_vars, A, pi, ll = fit_hmm(sc.values, n_states=6, seed=seed)
                ari_scores.append(adjusted_rand_score(states, hmm_viterbi(sc.values, means, log_vars, A, pi)))
        silhouette = float("nan")
        if len(np.unique(states)) > 1:
            silhouette = float(silhouette_score(sc.values, states.values))
        diagnostic_rows.append(
            {
                "engine": eng,
                "config": name,
                "n_feat": len(feats_used),
                "n_PCs": sc.shape[1],
                "n_states": 6,
                "silhouette": round(silhouette, 3) if np.isfinite(silhouette) else np.nan,
                "BIC": round(float(bic), 1),
                "ari_mean": round(float(np.mean(ari_scores)), 3),
                "ari_std": round(float(np.std(ari_scores)), 3),
            }
        )

diag_df = pd.DataFrame(diagnostic_rows)
diag_df.to_csv(REGIME_OUT / "table_diagnostics.csv", index=False)
diag_df.to_csv(REGIME_OUT / "table_diagnostics_summary.csv", index=False)

print("Metrics table:")
display(metrics_df)

# %% [markdown]
# ## 3. Final HMM Model and YC-Compatible State Export
#
# The final monthly HMM export is the bridge consumed by the allocation stage.
# It carries the state path, crisis state, and monthly posterior probabilities
# needed by the downstream CVaR workflow.

# %%
print("=== Final HMM model ===", flush=True)
feats_f = bn_select(fb_alt, fb_macro)
scores_f, feats_f, pca_f = subset_pca(fb_alt, feats_f)
Xf = scores_f.values
means_f, log_vars_f, A_f, pi_f, ll_f = fit_hmm(Xf, n_states=6, seed=0)
states_f = pd.Series(hmm_viterbi(Xf, means_f, log_vars_f, A_f, pi_f), index=scores_f.index, name="state")
probs_f = pd.DataFrame(hmm_filter(Xf, means_f, log_vars_f, A_f, pi_f), index=scores_f.index, columns=[f"p_state{k}" for k in range(6)])
crisis_f = ownership_crisis(states_f.values, Xf)

final_feature_rows = []
Xfeat = fb_alt.transformed.loc[scores_f.index, feats_f].astype(float)
Xraw = alt_all[feats_f].reindex(scores_f.index).astype(float)
Zfeat = (Xfeat - Xfeat.mean()) / Xfeat.std(ddof=0).replace(0, 1.0)
for s in sorted(states_f.unique()):
    mask = states_f == s
    for c in feats_f:
        final_feature_rows.append(
            {
                "state": int(s),
                "is_crisis": int(s == crisis_f),
                "feature": c,
                "description": FEATURE_DESC.get(c, ""),
                "n_months": int(mask.sum()),
                "mean_z": round(float(Zfeat.loc[mask, c].mean()), 3),
                "mean_transformed": round(float(Xfeat.loc[mask, c].mean()), 4),
                "mean_untransformed": round(float(Xraw.loc[mask, c].mean()), 4),
            }
        )
final_feature_df = pd.DataFrame(final_feature_rows)
final_feature_df.to_csv(REGIME_OUT / "final_feature_by_state.csv", index=False)

load = pd.DataFrame(pca_f.components_.T, index=feats_f, columns=[f"PC{i+1}" for i in range(scores_f.shape[1])])
evr = pd.Series(pca_f.explained_variance_ratio_, index=load.columns, name="explained_var_ratio")
out_load = load.round(4).copy()
out_load.loc["__explained_var_ratio__"] = evr.round(4)
out_load.loc["__cum_var_ratio__"] = evr.cumsum().round(4)
out_load.to_csv(REGIME_OUT / "final_pc_loadings.csv")

pc_mean = scores_f.groupby(states_f).mean().T
pc_mean.columns = [f"state{c}" for c in pc_mean.columns]
pc_mean.to_csv(REGIME_OUT / "final_pc_by_state.csv")

trans_f = pd.DataFrame(A_f, index=[f"from_state{i}" for i in range(6)], columns=[f"to_state{j}" for j in range(6)])
trans_f.to_csv(REGIME_OUT / "final_transition_matrix.csv")

final_prob = probs_f.copy()
final_prob.insert(0, "state", states_f.values)
final_prob.insert(1, "crisis", (states_f.values == crisis_f).astype(int))
final_prob.insert(2, "p_crisis", probs_f.iloc[:, crisis_f].values)
final_prob.index.name = "date"
final_prob = final_prob.reset_index()
final_prob["date"] = pd.to_datetime(final_prob["date"]).dt.strftime("%Y-%m-%d")
final_prob.to_csv(REGIME_OUT / "final_state_probabilities.csv", index=False)

yc_state_file = REGIME_OUT / "states_altdata_bnmeanrank_hmm_walkforward.csv"
yc_state_contract = final_prob.copy()
yc_state_contract.to_csv(yc_state_file, index=False)

fig1, ax1 = plt.subplots(figsize=(12, 4))
im = ax1.imshow(Zfeat.groupby(states_f).mean().values, aspect="auto", cmap="RdYlGn")
ax1.set_yticks(range(len(sorted(states_f.unique()))))
ax1.set_yticklabels([f"state{i}" for i in sorted(states_f.unique())])
ax1.set_xticks(range(len(feats_f)))
ax1.set_xticklabels([f"{c}" for c in feats_f], rotation=45, ha="right")
fig1.colorbar(im, ax=ax1)
fig1.tight_layout()
fig1.savefig(REGIME_OUT / "fig_feature_by_state_z.png", dpi=140, bbox_inches="tight")
plt.close(fig1)

fig2, ax2 = plt.subplots(figsize=(10, 4))
im = ax2.imshow(load.values, aspect="auto", cmap="RdYlBu")
ax2.set_yticks(range(len(feats_f)))
ax2.set_yticklabels(feats_f)
ax2.set_xticks(range(len(load.columns)))
ax2.set_xticklabels(load.columns, rotation=45, ha="right")
fig2.colorbar(im, ax=ax2)
fig2.tight_layout()
fig2.savefig(REGIME_OUT / "fig_pc_loadings.png", dpi=140, bbox_inches="tight")
plt.close(fig2)

fig3, ax3 = plt.subplots(figsize=(10, 4))
im = ax3.imshow(pc_mean.values, aspect="auto", cmap="RdYlBu")
ax3.set_yticks(range(len(pc_mean.index)))
ax3.set_yticklabels(pc_mean.index)
ax3.set_xticks(range(len(pc_mean.columns)))
ax3.set_xticklabels(pc_mean.columns, rotation=45, ha="right")
fig3.colorbar(im, ax=ax3)
fig3.tight_layout()
fig3.savefig(REGIME_OUT / "fig_pc_by_state.png", dpi=140, bbox_inches="tight")
plt.close(fig3)

fig4, ax4 = plt.subplots(figsize=(5, 4))
im = ax4.imshow(A_f, cmap="viridis", vmin=0, vmax=1)
ax4.set_xticks(range(6))
ax4.set_yticks(range(6))
ax4.set_xlabel("to state")
ax4.set_ylabel("from state")
fig4.colorbar(im, ax=ax4)
fig4.tight_layout()
fig4.savefig(REGIME_OUT / "fig_transition_matrix.png", dpi=140, bbox_inches="tight")
plt.close(fig4)

print("Final model selected features:", feats_f)
print("Crisis state:", crisis_f)
print("Final state probabilities saved to:", REGIME_OUT)

# %% [markdown]
# ## 4. CVaR Gamma-Grid Evaluation
#
# This section reads the root ETF returns together with the integrated monthly
# state export and evaluates the four allocation strategies across the paper
# gamma grid.

# %%
print("=== CVaR evaluation ===", flush=True)


def load_etf_returns() -> pd.DataFrame:
    ret = pd.read_csv(FILES["etf_returns"])
    date_col = "month" if "month" in ret.columns else ret.columns[0]
    ret[date_col] = pd.to_datetime(ret[date_col]).dt.to_period("M").dt.to_timestamp()
    ret = ret.set_index(date_col).sort_index()
    ret = ret.apply(pd.to_numeric, errors="coerce").dropna(how="any")
    return ret


def load_regime_file(path: Path):
    df = pd.read_csv(path)
    if "date" in df.columns:
        date_col = "date"
    elif "sasdate" in df.columns:
        date_col = "sasdate"
    elif "month" in df.columns:
        date_col = "month"
    else:
        date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col]).dt.to_period("M").dt.to_timestamp()
    df = df.set_index(date_col).sort_index()
    state_cols = [c for c in df.columns if c.startswith("p_state")]
    probs = df[state_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    probs.columns = [int(c.replace("p_state_", "").replace("p_state", "")) for c in probs.columns]
    probs = probs.reindex(sorted(probs.columns), axis=1)
    probs = probs.div(probs.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(1.0 / len(probs.columns))
    out = pd.DataFrame(index=df.index)
    out["state"] = pd.to_numeric(df["state"], errors="coerce").astype("Int64")
    out["crisis"] = pd.to_numeric(df.get("crisis", (out["state"] == crisis_f).astype(int)), errors="coerce")
    if "p_crisis" in df.columns:
        out["p_crisis"] = pd.to_numeric(df["p_crisis"], errors="coerce")
    else:
        out["p_crisis"] = probs.get(crisis_f, probs.iloc[:, 0])
    return out, probs


returns_etf = load_etf_returns()
regime_labels_df, regime_probs_df = load_regime_file(yc_state_file)
asset_cols = returns_etf.columns.tolist()
n_assets = len(asset_cols)


def theta_from_gamma(gamma: float, n_obs: int, n_assets: int) -> float:
    return float(gamma * max(n_obs, 1) ** (-1.0 / max(n_assets, 1)))


def estimate_transition_matrix(states, state_values, smoothing=1e-3) -> pd.DataFrame:
    states = pd.Series(states).dropna().astype(int)
    k = len(state_values)
    idx = {s: i for i, s in enumerate(state_values)}
    mat = np.full((k, k), smoothing, dtype=float)
    vals = states.values
    for a, b in zip(vals[:-1], vals[1:]):
        if a in idx and b in idx:
            mat[idx[a], idx[b]] += 1.0
    mat = mat / mat.sum(axis=1, keepdims=True)
    return pd.DataFrame(mat, index=state_values, columns=state_values)


def next_regime_weights(prev_month, probs, train_states, state_values):
    pi = probs.loc[prev_month].reindex(state_values).fillna(0.0)
    pi = pi / pi.sum() if pi.sum() > 0 else pd.Series(1.0 / len(state_values), index=state_values)
    A = estimate_transition_matrix(train_states, state_values)
    w = pi.values @ A.loc[pi.index, state_values].values
    out = pd.Series(w, index=state_values).clip(lower=0.0)
    return out / out.sum(), A


def observation_probs_from_regime_weights(train_states, regime_weights):
    train_states = pd.Series(train_states).astype(int)
    obs_probs = pd.Series(0.0, index=train_states.index, dtype=float)
    missing_weight = 0.0
    for k, wk in regime_weights.items():
        mask = train_states == int(k)
        count = int(mask.sum())
        if count > 0:
            obs_probs.loc[mask] = float(wk) / count
        else:
            missing_weight += float(wk)
    if obs_probs.sum() <= 0:
        obs_probs[:] = 1.0 / len(obs_probs)
    else:
        if missing_weight > 0:
            obs_probs += missing_weight / len(obs_probs)
        obs_probs = obs_probs / obs_probs.sum()
    return obs_probs.values


def regime_weighted_theta(gamma, train_states, regime_weights, n_assets):
    train_states = pd.Series(train_states).astype(int)
    theta = 0.0
    for k, wk in regime_weights.items():
        nk = int((train_states == int(k)).sum())
        if nk > 0 and wk > 0:
            theta += float(wk) * theta_from_gamma(gamma, nk, n_assets)
    return float(theta)


def effective_n_assets(weights):
    weights = np.asarray(weights, dtype=float)
    denom = np.sum(np.square(weights))
    return 1.0 / denom if denom > 0 else np.nan


def solve_weighted_cvar_lp_l1(R, obs_probs, eta, theta_eff=0.0, max_weight=1.0):
    R = np.asarray(R, dtype=float)
    n, i_assets = R.shape
    obs_probs = np.asarray(obs_probs, dtype=float)
    obs_probs = np.clip(obs_probs, 0.0, None)
    obs_probs = obs_probs / obs_probs.sum()

    ix_x = 0
    ix_v = i_assets
    ix_u = i_assets + 1
    ix_z = i_assets + 1 + n
    n_var = i_assets + 1 + n + 1

    c = np.zeros(n_var)
    c[ix_v] = 1.0
    c[ix_u:ix_u + n] = obs_probs / (1.0 - eta)
    c[ix_z] = theta_eff / (1.0 - eta)

    A_ub = []
    b_ub = []
    for row_i in range(n):
        row = np.zeros(n_var)
        row[ix_x:ix_x + i_assets] = -R[row_i]
        row[ix_v] = -1.0
        row[ix_u + row_i] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)
    for j in range(i_assets):
        row = np.zeros(n_var)
        row[ix_x + j] = 1.0
        row[ix_z] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)

    A_eq = np.zeros((1, n_var))
    A_eq[0, ix_x:ix_x + i_assets] = 1.0
    b_eq = np.array([1.0])
    bounds = [(0.0, max_weight)] * i_assets + [(None, None)] + [(0.0, None)] * n + [(0.0, max_weight)]

    res = linprog(
        c,
        A_ub=np.asarray(A_ub),
        b_ub=np.asarray(b_ub),
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not res.success:
        x = np.ones(i_assets) / i_assets
        loss = -R @ x
        v = float(np.quantile(loss, eta))
        u = np.maximum(loss - v, 0.0)
        obj = float(v + np.sum(obs_probs * u) / (1.0 - eta) + theta_eff * np.max(x) / (1.0 - eta))
        return {"success": False, "message": res.message, "x": x, "v": v, "z": float(np.max(x)), "objective": obj}

    x = res.x[ix_x:ix_x + i_assets]
    v = float(res.x[ix_v])
    z = float(res.x[ix_z])
    return {"success": True, "message": res.message, "x": x, "v": v, "z": z, "objective": float(res.fun)}


def max_drawdown(ret):
    ret = pd.Series(ret).dropna()
    if ret.empty:
        return np.nan
    wealth = (1.0 + ret).cumprod()
    peak = wealth.cummax()
    return float((wealth / peak - 1.0).min())


def empirical_var_cvar(ret, eta=0.95):
    ret = pd.Series(ret).dropna()
    if ret.empty:
        return np.nan, np.nan
    losses = -ret
    var = float(np.quantile(losses, eta))
    tail = losses[losses >= var]
    cvar = float(tail.mean()) if len(tail) else var
    return var, cvar


def performance_metrics(ret, periods_per_year=12):
    ret = pd.Series(ret).dropna()
    if ret.empty:
        return {
            "n_months": 0,
            "ann_return": np.nan,
            "ann_vol": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "VaR_95_loss": np.nan,
            "CVaR_95_loss": np.nan,
            "final_wealth": np.nan,
        }
    wealth = (1.0 + ret).cumprod()
    ann_return = float(wealth.iloc[-1] ** (periods_per_year / len(ret)) - 1.0)
    ann_vol = float(ret.std(ddof=1) * np.sqrt(periods_per_year))
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
    var95, cvar95 = empirical_var_cvar(ret, 0.95)
    return {
        "n_months": int(len(ret)),
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(ret),
        "VaR_95_loss": var95,
        "CVaR_95_loss": cvar95,
        "final_wealth": float(wealth.iloc[-1]),
    }


all_returns = []
all_weights = []
all_diagnostics = []

R_all = returns_etf.copy()

for gamma in GAMMA_GRID:
    print(f"Running gamma={gamma:.3f}", flush=True)
    common = R_all.index.intersection(regime_labels_df.index).intersection(regime_probs_df.index)
    R = R_all.loc[common]
    labels = regime_labels_df.loc[common]
    probs = regime_probs_df.loc[common]
    state_values = sorted(probs.columns.tolist())
    if len(R) <= TRAIN_WINDOW:
        continue
    for i in range(TRAIN_WINDOW, len(R), REBALANCE_STEP):
        rebalance_month = R.index[i]
        prev_month = R.index[i - 1]
        train_R = R.iloc[i - TRAIN_WINDOW:i]
        test_r = R.iloc[i]
        train_states = labels["state"].iloc[i - TRAIN_WINDOW:i].astype(int)
        n_train = len(train_R)
        equal_obs_probs = np.ones(n_train) / n_train
        theta_dr = theta_from_gamma(gamma, n_train, n_assets)
        regime_w, A_rolling = next_regime_weights(prev_month, probs, train_states, state_values)
        rs_obs_probs = observation_probs_from_regime_weights(train_states, regime_w)
        theta_rsdr = regime_weighted_theta(gamma, train_states, regime_w, n_assets)
        strategy_specs = {
            "SAA-CVaR": {"obs_probs": equal_obs_probs, "theta": 0.0},
            "DR-CVaR": {"obs_probs": equal_obs_probs, "theta": theta_dr},
            "RS-CVaR": {"obs_probs": rs_obs_probs, "theta": 0.0},
            "RSDR-CVaR": {"obs_probs": rs_obs_probs, "theta": theta_rsdr},
        }
        for strategy, spec in strategy_specs.items():
            sol = solve_weighted_cvar_lp_l1(
                train_R.values,
                spec["obs_probs"],
                ETA,
                theta_eff=spec["theta"],
                max_weight=1.0,
            )
            w = sol["x"]
            realized_return = float(np.dot(w, test_r.values))
            predicted_state = int(labels["state"].loc[rebalance_month])
            predicted_state_prob = float(probs.loc[prev_month].max())
            all_returns.append(
                {
                    "gamma": gamma,
                    "regime_source": SELECTED_REGIME_SOURCE,
                    "month": rebalance_month,
                    "strategy": strategy,
                    "return": realized_return,
                    "predicted_state": predicted_state,
                }
            )
            row = {
                "gamma": gamma,
                "regime_source": SELECTED_REGIME_SOURCE,
                "rebalance_month": rebalance_month,
                "strategy": strategy,
                "success": sol["success"],
                "objective": sol["objective"],
                "VaR_v": sol["v"],
                "theta_used": spec["theta"],
                "robust_norm_value": sol["z"],
                "robust_penalty_component": spec["theta"] * sol["z"] / (1.0 - ETA),
                "max_weight": float(np.max(w)),
                "effective_n_assets": float(effective_n_assets(w)),
                "predicted_state": predicted_state,
                "predicted_state_prob": predicted_state_prob,
                "regime_weight_max": float(regime_w.max()),
                "regime_weight_argmax": int(regime_w.idxmax()),
                "message": sol["message"],
            }
            row.update({asset: float(weight) for asset, weight in zip(asset_cols, w)})
            all_weights.append(row)
            all_diagnostics.append(row.copy())

strategy_returns = pd.DataFrame(all_returns)
strategy_weights = pd.DataFrame(all_weights)
diagnostics = pd.DataFrame(all_diagnostics)

metric_rows = []
turnover_rows = []
for (gamma, regime_source), g in strategy_returns.groupby(["gamma", "regime_source"]):
    pivot = g.pivot(index="month", columns="strategy", values="return").sort_index()
    pivot["Equal_Weight_ETF"] = returns_etf.loc[pivot.index, asset_cols].mean(axis=1)
    if "SPY" in returns_etf.columns:
        pivot["SPY_Benchmark"] = returns_etf.loc[pivot.index, "SPY"]
    for strategy in pivot.columns:
        row = {"gamma": gamma, "regime_source": regime_source, "strategy": strategy}
        row.update(performance_metrics(pivot[strategy]))
        metric_rows.append(row)
    wg = strategy_weights[(strategy_weights["gamma"] == gamma) & (strategy_weights["regime_source"] == regime_source)]
    for strategy, sg in wg.groupby("strategy"):
        w = sg.sort_values("rebalance_month")[asset_cols]
        turnover = w.diff().abs().sum(axis=1).dropna()
        turnover_rows.append(
            {
                "gamma": gamma,
                "regime_source": regime_source,
                "strategy": strategy,
                "avg_turnover": float(turnover.mean()) if len(turnover) else np.nan,
                "max_turnover": float(turnover.max()) if len(turnover) else np.nan,
            }
        )

metrics = pd.DataFrame(metric_rows)
turnover = pd.DataFrame(turnover_rows)
metrics = metrics.merge(turnover, on=["gamma", "regime_source", "strategy"], how="left")

strategy_returns.to_csv(CVAR_OUT / "etf_strategy_monthly_returns_gamma_grid.csv", index=False)
strategy_weights.to_csv(CVAR_OUT / "etf_strategy_weights_gamma_grid.csv", index=False)
diagnostics.to_csv(CVAR_OUT / "etf_optimizer_diagnostics_gamma_grid.csv", index=False)
metrics.to_csv(CVAR_OUT / "etf_performance_metrics_gamma_grid.csv", index=False)

for (gamma, regime_source), g in strategy_returns.groupby(["gamma", "regime_source"]):
    pivot = g.pivot(index="month", columns="strategy", values="return").sort_index()
    pivot["Equal_Weight_ETF"] = returns_etf.loc[pivot.index, asset_cols].mean(axis=1)
    if "SPY" in returns_etf.columns:
        pivot["SPY_Benchmark"] = returns_etf.loc[pivot.index, "SPY"]
    wealth = (1.0 + pivot).cumprod()
    plt.figure(figsize=(11, 5))
    for col in wealth.columns:
        lw = 2.5 if "RSDR" in col else 1.4
        plt.plot(wealth.index, wealth[col], label=col, linewidth=lw)
    plt.title(f"Cumulative Wealth: {regime_source}, gamma={gamma}")
    plt.ylabel("Growth of $1")
    plt.xlabel("Month")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(CVAR_OUT / f"cumulative_wealth_gamma_{str(gamma).replace('.', 'p')}_{regime_source}.png", dpi=160, bbox_inches="tight")
    plt.close()

print("CVaR outputs saved to:", CVAR_OUT)

print("=== Integrated output summary ===", flush=True)
for label, folder, extras in [
    ("root_gmm", ROOT_GMM_OUT, [ROOT_GMM_FIG]),
    ("regime_states", REGIME_OUT, []),
    ("cvar", CVAR_OUT, []),
]:
    print(f"[{label}] {folder}")
    for name in sorted(p.name for p in folder.iterdir() if p.is_file())[:12]:
        print(f"  - {name}")
    for extra in extras:
        if extra.exists():
            for name in sorted(p.name for p in extra.iterdir() if p.is_file())[:6]:
                print(f"  - {extra.name}/{name}")
    print()

# %% [markdown]
# ## 5. Run Summary
#
# The integrated pipeline keeps the active artefacts in the root
# `outputs/integrated/` tree:
#
# - `root_gmm/`
# - `regime_states/`
# - `cvar/`
#
# The integrated workflow writes outputs to:
#
# - `outputs/integrated/root_gmm/`
# - `outputs/integrated/regime_states/`
# - `outputs/integrated/cvar/`
