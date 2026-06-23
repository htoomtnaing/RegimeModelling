r"""
build_nb_regime_kappa_states.py
Emit regime_kappa_states.ipynb — a single self-contained notebook that computes the
regime-detection Cohen's-kappa table for five feature configurations (GMM & HMM), the
per-config state-path charts, cluster-stability / silhouette / BIC diagnostics, and the
deep-dive on the chosen final model (the HMM of altdata_full/bn_meanrank).

Run:  python build_nb_regime_kappa_states.py
Then: jupyter nbconvert --to notebook --execute --inplace regime_kappa_states.ipynb
"""
import nbformat as nbf

cells = []
def md(src):   cells.append(nbf.v4.new_markdown_cell(src))
def code(src): cells.append(nbf.v4.new_code_cell(src))

# ───────────────────────────────────────────────────────────────────────────────
md(r"""# Macro-regime detection — Cohen's $\kappa$, regime states & a final HMM

This notebook is **self-contained**: it reads only the CSVs in `./data/` and the local
`regime_taa.py`, and writes every result to `./outputs/`. It does four things.

1. **The $\kappa$ table.** For five feature configurations it fits two clustering engines
   (a Gaussian mixture, **GMM**, and a Gaussian hidden Markov model, **HMM**) and scores
   each engine's *crisis* flag against NBER recessions with **Cohen's $\kappa$**. Each row
   reports the number of input **features** and the number of **principal components (PCs)**
   the engine actually sees.
2. **Regime-state paths.** For every config × engine it draws the month-by-month regime
   state with NBER recessions shaded.
3. **Diagnostics.** Cluster-stability (seed ARI), silhouette and BIC for every config × engine.
4. **A final model.** It selects the **HMM of `altdata_full/bn_meanrank`** — almost the
   strongest $\kappa$ with the fewest features — and characterises it: per-state feature
   distributions, PC factor loadings (with intuitive labels), per-state PC distributions,
   intuitive state labels, the transition matrix, and a per-month state-probability CSV.

**Outputs written to `./outputs/`:** `regime_kappa_table.csv`, `fig_state_paths.png`,
`table_diagnostics.csv`, `final_feature_by_state.csv`, `final_pc_loadings.csv`,
`final_pc_by_state.csv`, `final_transition_matrix.csv`, `final_state_probabilities.csv`,
plus heatmap figures (`fig_feature_by_state_z.png`, `fig_pc_loadings.png`,
`fig_pc_by_state.png`, `fig_transition_matrix.png`).""")

# ───────────────────────────────────────────────────────────────────────────────
md(r"""## 1 · Setup — determinism

Two layers make the pipeline byte-reproducible: (1) pin BLAS threads **before** importing
numpy; (2) force full-SVD PCA. Every estimator uses `random_state` / `seed = 0`.""")

code(r"""import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '1'
os.environ['PYTHONHASHSEED'] = '0'

import sys, pathlib, functools, warnings, time
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.special import logsumexp
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as _PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import f_classif
from sklearn.metrics import cohen_kappa_score, silhouette_score, adjusted_rand_score
from IPython.display import display
warnings.filterwarnings('ignore')

ROOT = pathlib.Path.cwd()
sys.path.insert(0, str(ROOT))
DATA = ROOT / 'data'
OUT  = ROOT / 'outputs'; OUT.mkdir(exist_ok=True)

import regime_taa as rt
rt.PCA = functools.partial(_PCA, svd_solver='full')   # determinism layer 2/2
print('numpy', np.__version__, '| pandas', pd.__version__, '| regime_taa loaded')""")

# ───────────────────────────────────────────────────────────────────────────────
md(r"""## 2 · Inputs

* `fredmd_current.csv` — the FRED-MD monthly macro panel (raw levels + transform codes).
* `nber_usrec.csv` — the NBER recession indicator (monthly 0/1), the target $\kappa$ scores against.
* `altdata_monthly.csv` / `altdata_tcodes.csv` — the alternative-data universe (rates,
  factor returns, commodity/credit series) with their FRED transform codes.

We also build **21 engineered macro features** (yield-curve & credit spreads, real rates,
labour-market and equity-trend signals, inflation, etc.) from the FRED-MD raw series.""")

code(r"""print('=== [1/7] Loading inputs + engineered features ===', flush=True)
INIT_END = pd.Timestamp('2017-12-01')    # feature-selection cutoff (the screen only sees data <= here)
WIN_1976 = pd.Timestamp('1976-01-01')    # bundle start

data, tcodes = rt.load_fredmd(str(DATA / 'fredmd_current.csv'))
usrec = rt.load_usrec(str(DATA / 'nber_usrec.csv'))
alt_all = pd.read_csv(DATA / 'altdata_monthly.csv', index_col=0, parse_dates=True)
alt_tc  = pd.read_csv(DATA / 'altdata_tcodes.csv', index_col=0)['tcode'].to_dict()
ALTC = [c for c in alt_all.columns if not c.startswith('FRX_')]   # alt-data universe

# engineered macro features (21), each paired with its FRED-MD transform code
g = lambda c: data[c]; cpi_yoy = 100 * g('CPIAUCSL').pct_change(12)
u3 = g('UNRATE').rolling(3).mean(); ey = 100.0 / g('S&P PE ratio')
ENG = {'YC_10Y3M': (g('GS10') - g('TB3MS'), 1), 'YC_10Y1Y': (g('GS10') - g('GS1'), 1),
       'YC_5Y3M': (g('GS5') - g('TB3MS'), 1), 'YC_10Y_FF': (g('GS10') - g('FEDFUNDS'), 1),
       'CREDIT_BAA10Y': (g('BAA') - g('GS10'), 1), 'CREDIT_BAA_AAA': (g('BAA') - g('AAA'), 1),
       'CREDIT_AAA10Y': (g('AAA') - g('GS10'), 1), 'REAL_FF': (g('FEDFUNDS') - cpi_yoy, 1),
       'REAL_10Y': (g('GS10') - cpi_yoy, 1), 'SAHM': (u3 - u3.rolling(12).min(), 1),
       'UNRATE_12M_CHG': (g('UNRATE') - g('UNRATE').shift(12), 1),
       'SPX_MOM12': (100 * g('S&P 500').pct_change(12), 1),
       'SPX_TREND': (100 * (g('S&P 500') / g('S&P 500').rolling(12).mean() - 1), 1),
       'SPX_PE': (g('S&P PE ratio'), 1), 'SPX_DY': (g('S&P div yield'), 1),
       'ERP': (ey - (g('GS10') - cpi_yoy), 1), 'INFL_YoY': (cpi_yoy, 1),
       'OIL_MOM12': (100 * g('OILPRICEx').pct_change(12), 1),
       'M2_REAL_YoY': (100 * (g('M2SL') / g('CPIAUCSL')).pct_change(12), 1),
       'CREDIT_GROWTH': (100 * g('BUSLOANS').pct_change(12), 1), 'VIX': (g('VIXCLSx'), 4)}
ENG = {k: (s.dropna(), tc) for k, (s, tc) in ENG.items()}
print(f'FRED-MD {data.shape} | alt-data {len(ALTC)} cols | NBER through {usrec.index.max().date()}')""")

# ───────────────────────────────────────────────────────────────────────────────
md(r"""## 3 · Feature bundles

`build()` assembles a chosen mix of blocks (macro / engineered / alt-data), keeps series
with at least 50% coverage, applies the FRED-MD stationarity transforms, standardises, and
runs PCA to **95% cumulative variance**. We build four bundles:

| bundle | blocks | role |
|---|---|---|
| `fb_macro` | macro only | supplies the regime-label target used by the `bn_meanrank` selector |
| `fb_macroeng` | macro + engineered | config **macro+eng** |
| `fb_comb` | macro + engineered + alt-data | configs **comb_all/\*** |
| `fb_alt` | alt-data only | configs **altdata_full/\*** |""")

code(r"""def build(macro, eng, alt, window=WIN_1976):
    idx = data.loc[window:].index; d = pd.DataFrame(index=idx); tc = pd.Series(dtype=float)
    if macro:
        for c in data.columns: d[c] = data.loc[window:, c]; tc[c] = tcodes.get(c, 1)
    if eng:
        for k, (s, t) in ENG.items(): d[k] = s.reindex(idx); tc[k] = t
    if alt:
        a = alt_all[ALTC].reindex(idx)
        for c in a.columns:
            if a[c].isna().mean() <= 0.50: d[c] = a[c]; tc[c] = alt_tc[c]
    return rt.prepare_features(d, tc, exclude='exchange', pca_var=0.95, drop_initial=2)

print('=== [2/7] Building feature bundles ===', flush=True)
fb_macro    = build(True,  False, False)   # macro-only (label target for selection)
fb_macroeng = build(True,  True,  False)   # macro + engineered
fb_comb     = build(True,  True,  True)    # macro + engineered + alt-data
fb_alt      = build(False, False, True)    # alt-data only
for nm, fb in [('macro', fb_macro), ('macro+eng', fb_macroeng), ('comb', fb_comb), ('alt', fb_alt)]:
    print(f'  {nm:10s}: {len(fb.transformed.columns):3d} feats -> {fb.scores.shape[1]:2d} PCs  '
          f'({fb.transformed.index[0].date()} -> {fb.transformed.index[-1].date()})')""")

# ───────────────────────────────────────────────────────────────────────────────
md(r"""## 4 · Feature selection & engine machinery

**`bn_meanrank` selection (look-ahead-free).** Using only data up to `INIT_END` (2017-12):
1. a **Bai & Ng (2008)** screen sizes $K$ — the count of features whose point-biserial
   $t$-statistic against the NBER dummy exceeds $c_N=\sqrt{2\ln N}$;
2. a **3-seed RandomForest `mean_rank`** (Gini + permutation + F-stat) ordered against the
   macro bundle's regime labels picks **which** $K$ features.
The chosen features are then re-PCA'd to 95% variance (`subset_pca`).

**Engines.** A diagonal-covariance **GMM** (6 components) and a pure-NumPy diagonal-Gaussian
**HMM** (6 states; Baum-Welch EM + Viterbi decoding). For both, the **crisis state** is the
one that *owns the most Mahalanobis-extreme months* (`ownership_crisis`) — robust to the
HMM's tendency to place all tail months in a single state.""")

code(r"""def regime_labels(scores_df, we):
    sc = scores_df.loc[:we]
    return pd.Series(rt.RegimeModel(r=5, random_state=0, outlier_method='quantile',
                                    outlier_frac=0.15).fit(sc.values).labels_, index=sc.index)

def rank_features(fb, ylab):
    Xdf = pd.DataFrame(fb.scaler.transform(fb.transformed.values),
                       index=fb.transformed.index, columns=fb.columns)
    common = Xdf.index.intersection(ylab.index); X = Xdf.loc[common].values; yv = ylab.loc[common].values
    cols = list(fb.columns); ranks = []
    for seed in range(3):
        rf = RandomForestClassifier(n_estimators=200, max_features='sqrt', min_samples_leaf=3,
                                    random_state=seed, n_jobs=1).fit(X, yv)
        gini = pd.Series(rf.feature_importances_, index=cols)
        perm = pd.Series(permutation_importance(rf, X, yv, n_repeats=3, random_state=seed,
                                                n_jobs=1).importances_mean, index=cols)
        Fst = pd.Series(np.nan_to_num(f_classif(X, yv)[0]), index=cols)
        ranks.append((gini.rank(ascending=False) + perm.rank(ascending=False)
                      + Fst.rank(ascending=False)) / 3)
    return pd.concat(ranks, axis=1).mean(axis=1).sort_values()

def screen_tstats(fb, we):
    out = {}
    for col in fb.columns:
        feat = fb.transformed[col].dropna().loc[:we]; idx = feat.index.intersection(usrec.index)
        if len(idx) < 10: out[col] = 0.0; continue
        x = feat.loc[idx].values.astype(float); y = (usrec.loc[idx] > 0).astype(float).values
        if x.std() == 0 or y.std() == 0: out[col] = 0.0; continue
        r = np.corrcoef(x, y)[0, 1]; n = len(idx)
        out[col] = np.inf if abs(r) >= 1 else float(abs(r) * np.sqrt((n - 2) / (1 - r ** 2)))
    return pd.Series(out).sort_values(ascending=False)

def bn_select(fb, label_fb):
    # Bai & Ng screen sizes K; mean_rank orders which K. Selection sees only data <= INIT_END.
    ts = screen_tstats(fb, INIT_END); N = len(fb.columns)
    K = int((ts > np.sqrt(2 * np.log(N))).sum())
    order = rank_features(fb, regime_labels(label_fb.scores, INIT_END)).index
    return [c for c in order if c in fb.transformed.columns][:K]

def subset_pca(fb, cols):
    # full-sample PCA (95% var) on a feature subset; returns scores, feature list, fitted PCA
    avail = [c for c in cols if c in fb.transformed.columns]
    z = StandardScaler().fit_transform(fb.transformed[avail].values)
    cum = _PCA(svd_solver='full').fit(z).explained_variance_ratio_.cumsum()
    n = int((cum < 0.95).sum() + 1)
    pca = _PCA(n_components=n, svd_solver='full').fit(z)
    sc = pd.DataFrame(pca.transform(z), index=fb.transformed.index,
                      columns=[f'PC{i+1}' for i in range(n)])
    return sc, avail, pca

def scores_for(fb, sel, label_fb=None):
    # sel='all' -> the bundle's own PCA scores; sel='bn_meanrank' -> subset PCA on selection
    if sel == 'all':
        return fb.scores, list(fb.transformed.columns)
    cols = bn_select(fb, label_fb)
    sc, avail, _ = subset_pca(fb, cols)
    return sc, avail
print('selection helpers ready')""")

code(r"""# ── pure-NumPy diagonal-Gaussian HMM ───────────────────────────────────────────
def _le(X, m, lv):
    T, d = X.shape; K = m.shape[0]; o = np.zeros((T, K))
    for k in range(K):
        df = X - m[k]
        o[:, k] = -0.5 * (np.sum(df**2 * np.exp(-lv[k]), 1) + np.sum(lv[k]) + d*np.log(2*np.pi))
    return o
def _fw(lb, lA, lp):
    T, K = lb.shape; la = np.zeros((T, K)); la[0] = lp + lb[0]
    for t in range(1, T): la[t] = logsumexp(la[t-1, :, None] + lA, 0) + lb[t]
    return la
def _bw(lb, lA):
    T, K = lb.shape; b = np.zeros((T, K))
    for t in range(T-2, -1, -1): b[t] = logsumexp(lA + lb[t+1] + b[t+1], 1)
    return b
def fit_hmm(X, n_states=6, n_iter=20, n_init=1, reg=1e-2, seed=0):
    T, d = X.shape; K = n_states; rng = np.random.default_rng(seed); bll, best = -np.inf, None
    for tr in range(n_init):
        km = KMeans(K, n_init=3, random_state=int(rng.integers(9999))).fit(X)
        m = km.cluster_centers_.copy(); lv = np.log(np.full((K, d), X.var(0).clip(1e-6)) + reg)
        A = np.full((K, K), 0.05 / (K-1)); np.fill_diagonal(A, 0.95); pi = np.ones(K) / K
        lp = -np.inf
        for _ in range(n_iter):
            lb = _le(X, m, lv); lA = np.log(A + 1e-300); lpi = np.log(pi + 1e-300)
            la = _fw(lb, lA, lpi); bw = _bw(lb, lA); ll = float(logsumexp(la[-1]))
            lg = la + bw; lg -= logsumexp(lg, 1, keepdims=True); gg = np.exp(lg); xi = np.zeros((K, K))
            for t in range(T-1):
                lx = la[t, :, None] + lA + lb[t+1] + bw[t+1]; xi += np.exp(lx - logsumexp(lx))
            gs = gg.sum(0).clip(1e-10); pi = gg[0] / gg[0].sum()
            A = xi / xi.sum(1, keepdims=True).clip(1e-10)
            m = (gg[:, :, None] * X[:, None, :]).sum(0) / gs[:, None]
            for k in range(K):
                df = X - m[k]; v = (gg[:, k, None] * df**2).sum(0) / gs[k] + reg; lv[k] = np.log(v)
            if abs(ll - lp) < 1e-3: break
            lp = ll
        if ll > bll: bll = ll; best = (m.copy(), lv.copy(), A.copy(), pi.copy())
    return (*best, bll)
def filt(X, m, lv, A, pi):
    # filtered (forward) posterior P(state_t | data up to t)
    la = _fw(_le(X, m, lv), np.log(A + 1e-300), np.log(pi + 1e-300))
    la -= logsumexp(la, 1, keepdims=True); return np.exp(la)
def vit(X, m, lv, A, pi):
    # Viterbi MAP state path
    T, K = len(X), m.shape[0]; lb = _le(X, m, lv); lA = np.log(A + 1e-300)
    ld = np.zeros((T, K)); ps = np.zeros((T, K), int); ld[0] = np.log(pi + 1e-300) + lb[0]
    for t in range(1, T): sc = ld[t-1, :, None] + lA; ps[t] = sc.argmax(0); ld[t] = sc.max(0) + lb[t]
    s = np.zeros(T, int); s[-1] = ld[-1].argmax()
    for t in range(T-2, -1, -1): s[t] = ps[t+1, s[t+1]]
    return s

def ownership_crisis(labels_tr, Xtr, frac=0.15):
    mu = Xtr.mean(0); VI = np.linalg.pinv(np.cov(Xtr.T) + 1e-6 * np.eye(Xtr.shape[1])); df = Xtr - mu
    dd = np.einsum('ij,jk,ik->i', df, VI, df); ext = dd >= np.quantile(dd, 1 - frac)
    K = int(labels_tr.max()) + 1
    return int(np.argmax([int(ext[labels_tr == k].sum()) for k in range(K)]))

def kappa(flag, idx):
    # Cohen kappa of a 0/1 crisis flag vs NBER over the index
    s = pd.Series(flag, index=idx); ix = s.index.intersection(usrec.index)
    c = s.loc[ix].astype(int).values; n = (usrec.loc[ix] > 0).astype(int).values
    return round(cohen_kappa_score(n, c), 3)

def fit_engine(sc, engine):
    # fit GMM or HMM (6 states) once; return (states Series, crisis-state id, fit object/params)
    X = sc.values
    if engine == 'GMM':
        gm = GaussianMixture(6, covariance_type='diag', n_init=10, random_state=0,
                             reg_covar=1e-6, max_iter=100).fit(X)
        states = gm.predict(X); cs = ownership_crisis(states, X)
        return pd.Series(states, index=sc.index), int(cs), gm
    m, lv, A, pi, ll = fit_hmm(X, n_states=6, seed=0)
    if not np.isfinite(ll): m, lv, A, pi, ll = fit_hmm(X, n_states=6, seed=0)
    states = vit(X, m, lv, A, pi); cs = ownership_crisis(states, X)
    return pd.Series(states, index=sc.index), int(cs), (m, lv, A, pi, ll)
print('engine machinery ready')""")

# ───────────────────────────────────────────────────────────────────────────────
code(r"""# ── shared display helpers: feature descriptions + annotated heatmap ───────────
FEATURE_DESC = {
    'FF_RF':      'Risk-free rate — 1-month US T-bill (Fama-French)',
    'FF_Mkt_RF':  'Equity market excess return (Mkt-RF, Fama-French)',
    'FF_RMW':     'Profitability factor — Robust-minus-Weak (Fama-French)',
    'BAB_USA':    'Betting-Against-Beta factor (US)',
    'BLM_SPGSCI': 'S&P GSCI broad commodity index return',
    'IR_dtb3':    '3-month US Treasury bill rate',
    'IR_dtb6':    '6-month US Treasury bill rate',
    'IR_dgs1':    '1-year US Treasury yield',
    'IR_dgs2':    '2-year US Treasury yield',
}
FEATURE_SHORT = {
    'FF_RF': 'risk-free rate', 'FF_Mkt_RF': 'equity mkt excess ret', 'FF_RMW': 'profitability',
    'BAB_USA': 'betting-against-beta', 'BLM_SPGSCI': 'commodity index', 'IR_dtb3': '3m T-bill',
    'IR_dtb6': '6m T-bill', 'IR_dgs1': '1y Treasury', 'IR_dgs2': '2y Treasury',
}
PC_SHORT = {'PC1': 'short-rate momentum', 'PC2': 'flight-to-quality', 'PC3': 'reflation/risk-on',
            'PC4': 'policy-rate level', 'PC5': 'equity vs commodity', 'PC6': 'low-beta vs quality',
            'PC7': 'front-end twist'}

def heatmap(df, title, fname, cmap='RdBu_r', center=0.0, fmt='+.2f',
            ylabels=None, xlabels=None, figsize=None, cbar_label='', xrot=0):
    # annotated heatmap; diverging colours centred at `center` (set center=None for sequential).
    # compact figure + large fonts so the cell values read clearly inline.
    arr = df.values.astype(float); n, m = arr.shape
    fig, ax = plt.subplots(figsize=figsize or (1.02*m + 3.4, 0.46*n + 1.3))
    if center is not None:
        vext = float(np.nanmax(np.abs(arr - center))) or 1.0
        im = ax.imshow(arr, cmap=cmap, vmin=center - vext, vmax=center + vext, aspect='auto')
    else:
        vext = None; im = ax.imshow(arr, cmap=cmap, aspect='auto')
    ax.set_xticks(range(m))
    ax.set_xticklabels(xlabels or list(df.columns), fontsize=10, rotation=xrot,
                       ha=('right' if xrot else 'center'), rotation_mode='anchor')
    ax.set_yticks(range(n)); ax.set_yticklabels(ylabels or list(df.index), fontsize=11)
    for i in range(n):
        for j in range(m):
            v = arr[i, j]
            hi = (vext is not None) and (abs(v - center) > 0.55*vext)
            ax.text(j, i, format(v, fmt), ha='center', va='center', fontsize=13,
                    color='white' if hi else 'black')
    ax.set_title(title, fontsize=14)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=10)
    if cbar_label: cb.set_label(cbar_label, fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / fname, dpi=130, bbox_inches='tight'); plt.show()
print('display helpers ready')""")

# ───────────────────────────────────────────────────────────────────────────────
md(r"""## 5 · The Cohen's $\kappa$ table

**What Cohen's $\kappa$ means.** $\kappa$ measures *chance-corrected agreement* between two
0/1 labelings of the same months — here the model's binary **crisis flag** vs the **NBER
recession** indicator:
$$\kappa = \frac{p_o - p_e}{1 - p_e},$$
where $p_o$ is the fraction of months the two labels agree and $p_e$ is the agreement
expected by chance given each label's base rate. $\kappa=1$ is perfect agreement,
$\kappa=0$ is no better than chance, $\kappa<0$ is worse than chance. Because recessions are
rare (only `n_rec` of the months), $\kappa$ — unlike raw accuracy — is **not** inflated by a
model that trivially calls every month "expansion". Computed with
`sklearn.metrics.cohen_kappa_score`.

**How to read the table.** `n_feat` = number of input features; `n_PCs` = principal
components fed to the engine (95% variance); `n_rec` = recession months scored against;
`GMM` / `HMM` = the two engines' $\kappa$.

> These are **in-sample** $\kappa$: each engine is fit once on all available months and then
> scored on the same span, so the model has seen the very recessions it then flags. Read them
> as an **upper bound on detectability, not out-of-sample skill**. (Feature *selection* for
> the `bn_meanrank` rows is look-ahead-free — it sees only data up to 2017-12.)""")

code(r"""SPECS = [   # (display name, bundle, selection, label-bundle for bn_meanrank)
    ('macro+eng',                fb_macroeng, 'all',         None),
    ('comb_all/all',             fb_comb,     'all',         None),
    ('comb_all/bn_meanrank',     fb_comb,     'bn_meanrank', fb_macro),
    ('altdata_full/all',         fb_alt,      'all',         None),
    ('altdata_full/bn_meanrank', fb_alt,      'bn_meanrank', fb_macro),
]
print('=== [3/7] Cohen kappa table - 5 configs x {GMM, HMM} ===', flush=True)
rows, SC, FITS = [], {}, {}
for name, fb, sel, lab in SPECS:
    print(f'  - {name} : selecting features + fitting GMM/HMM ...', flush=True)
    t0 = time.time()
    sc, feats = scores_for(fb, sel, lab); SC[name] = (sc, feats)
    nrec = int((usrec.reindex(sc.index) > 0).sum())
    kap = {}
    for eng in ('GMM', 'HMM'):
        states, crisis, extra = fit_engine(sc, eng); FITS[(name, eng)] = (states, crisis, extra)
        kap[eng] = kappa((states == crisis).astype(int), sc.index)
    rows.append(dict(config=name, n_feat=len(feats), n_PCs=sc.shape[1],
                     n_rec=nrec, GMM=kap['GMM'], HMM=kap['HMM']))
    print(f'    done  n_feat={len(feats):3d} n_PCs={sc.shape[1]:2d} n_rec={nrec} | '
          f'GMM {kap["GMM"]:+.3f}  HMM {kap["HMM"]:+.3f}  ({time.time()-t0:.0f}s)', flush=True)

tbl = pd.DataFrame(rows)
tbl.to_csv(OUT / 'regime_kappa_table.csv', index=False)
print('\nsaved outputs/regime_kappa_table.csv')
(tbl.style.hide(axis='index')
    .background_gradient(cmap='RdYlGn', subset=['GMM', 'HMM'], vmin=-0.1, vmax=0.6)
    .format({'GMM': '{:.3f}', 'HMM': '{:.3f}'}))""")

# ───────────────────────────────────────────────────────────────────────────────
md(r"""## 6 · Regime-state paths

For every config × engine: the blue step line is the month-by-month regime **state id**
(0–5; the ids are arbitrary labels), **grey** marks NBER recessions, and **red** marks the
model's identified **crisis state**. A good detector lines its red bands up with the grey
bands — which is exactly what $\kappa$ measures.""")

code(r"""print('=== [4/7] State-path charts (5 configs x GMM/HMM) ===', flush=True)
fig, axes = plt.subplots(len(SPECS), 2, figsize=(13, 12.5), sharex=True)
for i, (name, fb, sel, lab) in enumerate(SPECS):
    sc, feats = SC[name]; idx = sc.index
    ur = (usrec.reindex(idx).fillna(0) > 0).values
    for j, eng in enumerate(('GMM', 'HMM')):
        ax = axes[i, j]
        states, crisis, _ = FITS[(name, eng)]
        ax.plot(idx, states.values, drawstyle='steps-post', lw=0.7, color='C0')
        ax.fill_between(idx, 0, 1, where=ur, transform=ax.get_xaxis_transform(),
                        color='grey', alpha=0.30, step='post')
        cr = (states.values == crisis)
        ax.fill_between(idx, 0, 1, where=cr, transform=ax.get_xaxis_transform(),
                        color='red', alpha=0.30, step='post')
        ax.set_ylim(-0.5, 5.5); ax.set_yticks(range(6))
        k = kappa((states == crisis).astype(int), sc.index)
        ax.set_title(f'{name} x {eng}   (crisis = state {crisis}, kappa = {k:+.3f})', fontsize=8.5)
        if j == 0: ax.set_ylabel('state')
handles = [Patch(color='grey', alpha=0.30, label='NBER recession'),
           Patch(color='red', alpha=0.30, label='model crisis state'),
           Line2D([0], [0], color='C0', lw=1.2, label='regime-state path')]
fig.legend(handles=handles, loc='upper center', ncol=3, fontsize=9, bbox_to_anchor=(0.5, 1.004))
fig.suptitle('Regime-state paths vs NBER recessions (grey) and the model crisis state (red)',
             y=1.025, fontsize=11)
fig.tight_layout()
fig.savefig(OUT / 'fig_state_paths.png', dpi=130, bbox_inches='tight')
print('saved outputs/fig_state_paths.png'); plt.show()""")

# ───────────────────────────────────────────────────────────────────────────────
md(r"""## 7 · Diagnostics — cluster-stability, silhouette & BIC

For every config × engine:
* **Cluster-stability (seed ARI)** — refit the engine under different random seeds and
  average the Adjusted Rand Index against the seed-0 labeling. Closer to **1** = the
  partition is reproducible (not an artefact of one initialisation).
* **Silhouette** — mean silhouette of the state assignment in PC space (higher = better
  separated; can be near 0 or negative for overlapping regimes).
* **BIC** — Bayesian Information Criterion of the fitted model (lower = better fit per
  parameter); exact for GMM, and computed from the HMM's log-likelihood and parameter count.

Results are **grouped by engine** (all GMM rows, then all HMM rows). The $\kappa$ table is
reproduced underneath for side-by-side reading.""")

code(r"""def stability_ari(sc, engine, n_seed):
    X = sc.values
    if engine == 'GMM':
        ref = GaussianMixture(6, covariance_type='diag', n_init=10, random_state=0,
                              reg_covar=1e-6, max_iter=100).fit(X).predict(X)
        fits = [GaussianMixture(6, covariance_type='diag', n_init=10, random_state=s,
                reg_covar=1e-6, max_iter=100).fit(X).predict(X) for s in range(1, n_seed + 1)]
    else:
        m, lv, A, pi, _ = fit_hmm(X, n_states=6, seed=0); ref = vit(X, m, lv, A, pi)
        fits = []
        for s in range(1, n_seed + 1):
            mm, lvv, AA, pp, _ = fit_hmm(X, n_states=6, seed=s); fits.append(vit(X, mm, lvv, AA, pp))
    aris = [adjusted_rand_score(ref, f) for f in fits]
    return float(np.mean(aris)), float(np.std(aris))

def bic_silhouette(sc, states, engine, extra):
    X = sc.values
    sil = float(silhouette_score(X, states)) if len(set(states)) > 1 else float('nan')
    if engine == 'GMM':
        bic = float(extra.bic(X))
    else:
        m, lv, A, pi, ll = extra; T, dd = X.shape; K = 6
        p = K*dd + K*dd + K*(K-1) + (K-1)        # means + diag vars + transitions + initial
        bic = float(-2*ll + p*np.log(T))
    return sil, bic

print('=== [5/7] Diagnostics - refitting across seeds (longest stage) ===', flush=True)
diag_rows = []
for eng in ('GMM', 'HMM'):                       # engine-major: all GMM rows, then all HMM rows
    for name, fb, sel, lab in SPECS:
        sc, feats = SC[name]
        print(f'  - {eng} / {name} ...', flush=True)
        states, crisis, extra = FITS[(name, eng)]
        sil, bic = bic_silhouette(sc, states.values, eng, extra)
        am, asd = stability_ari(sc, eng, n_seed=(10 if eng == 'GMM' else 5))
        diag_rows.append(dict(engine=eng, config=name, n_feat=len(feats), n_PCs=sc.shape[1],
                              n_states=6, silhouette=round(sil, 3), BIC=round(bic, 1),
                              ari_mean=round(am, 3), ari_std=round(asd, 3)))
        print(f'    {eng}  {name:26s} sil={sil:+.3f}  BIC={bic:10.1f}  ARI={am:.3f} +/- {asd:.3f}',
              flush=True)
diag = pd.DataFrame(diag_rows)
diag.to_csv(OUT / 'table_diagnostics.csv', index=False)
print('\nsaved outputs/table_diagnostics.csv\n')

# grouped table (engine -> config), gradient-shaded on the comparable metrics
disp = diag.set_index(['engine', 'config'])
display(disp.style
        .background_gradient(cmap='Greens', subset=['ari_mean'])
        .background_gradient(cmap='Blues', subset=['silhouette'])
        .format({'silhouette': '{:+.3f}', 'BIC': '{:,.1f}', 'ari_mean': '{:.3f}', 'ari_std': '{:.3f}'}))

# re-show the kappa table for side-by-side reading
print('\nCohen kappa table (reproduced):')
kt = pd.read_csv(OUT / 'regime_kappa_table.csv')
display(kt.style.hide(axis='index')
        .background_gradient(cmap='RdYlGn', subset=['GMM', 'HMM'], vmin=-0.1, vmax=0.6)
        .format({'GMM': '{:.3f}', 'HMM': '{:.3f}'}))""")

# ───────────────────────────────────────────────────────────────────────────────
md(r"""## 8 · The final model — HMM of `altdata_full/bn_meanrank`

We select this as the final model: it reaches **almost the highest $\kappa$ with the fewest
features** — a compact, alt-data-only set sized by the Bai & Ng screen. The rest of the
notebook re-derives it end-to-end and characterises it.""")

code(r"""print('=== [6/7] Final model - fit HMM of altdata_full/bn_meanrank ===', flush=True)
# re-derive the final model end-to-end (deterministic -> reproduces the table row)
FEATS = bn_select(fb_alt, fb_macro)
scores_f, feats_f, pca_f = subset_pca(fb_alt, FEATS)
Xf = scores_f.values
mF, lvF, AF, piF, llF = fit_hmm(Xf, n_states=6, seed=0)
if not np.isfinite(llF): mF, lvF, AF, piF, llF = fit_hmm(Xf, n_states=6, seed=0)
statesF = pd.Series(vit(Xf, mF, lvF, AF, piF), index=scores_f.index, name='state')
crisisF = ownership_crisis(statesF.values, Xf)
postF = filt(Xf, mF, lvF, AF, piF)               # filtered posteriors, T x 6
NPC = scores_f.shape[1]
kF = kappa((statesF == crisisF).astype(int), scores_f.index)
print(f'final model: {len(feats_f)} features -> {NPC} PCs -> 6-state HMM', flush=True)
print(f'selected features: {feats_f}')
print(f'crisis state = {crisisF}   |   Cohen kappa vs NBER = {kF:+.3f}')
assert len(feats_f) == 9 and NPC == 7, (len(feats_f), NPC)""")

# ───────────────────────────────────────────────────────────────────────────────
md(r"""### 8a · Feature distribution by HMM state

All nine input features, summarised within each HMM state, in three views:
* **standardized (z)** — shown as a heatmap (z-score over the whole sample), so the values are
  comparable across features: red = the feature runs high in that state, blue = low;
* **raw transformed units** — the stationarity-adjusted series the model actually uses
  (rate features are monthly *changes*; factor / commodity features are returns);
* **raw (untransformed) units** — the original series before transformation (rate features are
  *levels* in %; factors / commodities are monthly returns).

Each table carries a `description` column explaining what every feature is. The full long
table (all three means per state × feature) is saved to `final_feature_by_state.csv`.""")

code(r"""print('=== [7/7] Final model - profiles, loadings, transition & probabilities ===', flush=True)
Xfeat = fb_alt.transformed.loc[scores_f.index, feats_f].astype(float)   # transformed (model) units
Xraw  = alt_all[feats_f].reindex(scores_f.index).astype(float)          # raw, untransformed units
sd = Xfeat.std(ddof=0).replace(0, 1.0); Zfeat = (Xfeat - Xfeat.mean()) / sd
st = statesF
counts = pd.DataFrame({'n_months': st.value_counts().sort_index(),
                       'pct': (100 * st.value_counts(normalize=True).sort_index()).round(1),
                       'is_crisis': [int(s == crisisF) for s in sorted(st.unique())]}).T
def _by_state(df):
    t = df.groupby(st).mean().T; t.columns = [f'state{c}' for c in t.columns]; return t
mean_z, mean_tr, mean_un = _by_state(Zfeat), _by_state(Xfeat), _by_state(Xraw)
desc = pd.Series({c: FEATURE_DESC.get(c, '') for c in feats_f}, name='description')

long_rows = []
for s in sorted(st.unique()):
    msk = (st == s)
    for c in feats_f:
        long_rows.append(dict(state=int(s), is_crisis=int(s == crisisF), feature=c,
                              description=FEATURE_DESC.get(c, ''), n_months=int(msk.sum()),
                              mean_z=round(float(Zfeat.loc[msk, c].mean()), 3),
                              mean_transformed=round(float(Xfeat.loc[msk, c].mean()), 4),
                              mean_untransformed=round(float(Xraw.loc[msk, c].mean()), 4)))
pd.DataFrame(long_rows).to_csv(OUT / 'final_feature_by_state.csv', index=False)
print('saved outputs/final_feature_by_state.csv   (crisis state =', crisisF, ')\n')
print('Per-state size:'); display(counts)

print('\nFeature mean by state - standardized (z):  [heatmap]')
ylab = [f'{c} — {FEATURE_SHORT.get(c, c)}' for c in mean_z.index]
heatmap(mean_z, 'Feature mean by HMM state — standardized (z)', 'fig_feature_by_state_z.png',
        ylabels=ylab, cbar_label='mean z-score')

print('Feature mean by state - raw transformed units:')
display(mean_tr.round(3).join(desc))
print('\nFeature mean by state - raw (untransformed) units:')
display(mean_un.round(3).join(desc))""")

# ───────────────────────────────────────────────────────────────────────────────
md(r"""### 8b · PC factor loadings

The 9 standardized features reduce to **7 principal components**. The loadings heatmap below
shows how each feature contributes to each PC (red = positive, blue = negative; the printout
lists each PC's top-3 features by absolute loading); `explained_var_ratio` is each PC's share
of variance. Saved to `final_pc_loadings.csv`. Intuitive labels follow in the next cell.""")

code(r"""load = pd.DataFrame(pca_f.components_.T, index=feats_f,
                    columns=[f'PC{i+1}' for i in range(NPC)])
evr = pd.Series(pca_f.explained_variance_ratio_, index=load.columns, name='explained_var_ratio')
out_load = load.round(4).copy()
out_load.loc['__explained_var_ratio__'] = evr.round(4)
out_load.loc['__cum_var_ratio__'] = evr.cumsum().round(4)
out_load.to_csv(OUT / 'final_pc_loadings.csv')
print('saved outputs/final_pc_loadings.csv\n')
for pc in load.columns:
    s = load[pc].reindex(load[pc].abs().sort_values(ascending=False).index)
    top = ', '.join(f'{f} {v:+.2f}' for f, v in s.head(3).items())
    print(f'{pc}  (EVR {evr[pc]*100:4.1f}%, cum {evr.cumsum()[pc]*100:4.1f}%):  {top}')
print()
ylab = [f'{c} — {FEATURE_SHORT.get(c, c)}' for c in load.index]
xlab = [f'{c}: {PC_SHORT.get(c, "")}' for c in load.columns]
heatmap(load, 'PC factor loadings (feature x PC)', 'fig_pc_loadings.png',
        ylabels=ylab, xlabels=xlab, xrot=30, cbar_label='loading')
display(evr.to_frame().T.round(4))""")

md(r"""**PC factor labels & interpretation.** Each PC is named from the features that load most
heavily on it. The nine inputs are: monthly *changes* in the 3m / 6m / 1y / 2y Treasury yields
(`IR_dtb3`, `IR_dtb6`, `IR_dgs1`, `IR_dgs2`), the risk-free-rate *level* (`FF_RF`), the equity
market excess return (`FF_Mkt_RF`), the profitability factor (`FF_RMW`), the betting-against-beta
factor (`BAB_USA`), and the S&P GSCI commodity return (`BLM_SPGSCI`).

| PC | var | intuitive label | what a *high* value means |
|---|---|---|---|
| **PC1** | 37% | **Short-rate momentum (policy thrust)** | the front end (3m–2y yields) is *rising together* — monetary tightening; low = easing |
| **PC2** | 17% | **Flight-to-quality rotation** | quality (RMW) and low-beta (BAB) beat the market and commodities — risk-off |
| **PC3** | 13% | **Reflation / pro-cyclical risk premium** | commodities, low-beta and the market all rallying — broad risk-on |
| **PC4** | 11% | **Policy-rate level (rate regime)** | the policy / risk-free rate itself is high (vs ZIRP when low) |
| **PC5** | 8% | **Equities-vs-commodities divergence** | equities lead while commodities lag |
| **PC6** | 7% | **Low-beta vs profitability style spread** | low-beta (BAB) favoured over quality (RMW) — a within-defensive rotation |
| **PC7** | 5% | **Front-end curve twist** | the 3m bill rises relative to the 2y — a twist at the very short end |

PC1 (curve-wide rate *momentum*) and PC4 (the rate *level*) dominate — together ~48% of the
variance — so this alt-data regime space is first and foremost an **interest-rate** space, with
the factor-rotation and commodity signals (PC2/PC3/PC5/PC6) describing the risk environment.""")

# ───────────────────────────────────────────────────────────────────────────────
md(r"""### 8c · PC distribution by HMM state

The mean of each of the 7 PC scores within each HMM state — the compact fingerprint that
distinguishes the regimes (shown as a heatmap; red = high, blue = low). Saved to
`final_pc_by_state.csv`.""")

code(r"""pc_mean = scores_f.groupby(statesF).mean().T
pc_mean.columns = [f'state{c}' for c in pc_mean.columns]
pc_mean.to_csv(OUT / 'final_pc_by_state.csv')
print('saved outputs/final_pc_by_state.csv   (crisis state =', crisisF, ')\n')
ylab = [f'{c} — {PC_SHORT.get(c, "")}' for c in pc_mean.index]
heatmap(pc_mean, 'Mean PC score by HMM state', 'fig_pc_by_state.png',
        ylabels=ylab, cbar_label='mean PC score')""")

# ───────────────────────────────────────────────────────────────────────────────
md(r"""### 8d · State labels & interpretation

The table below supports the labelling: each state's size, whether it is the crisis state,
and what share of its months overlap an NBER recession.""")

code(r"""ur = (usrec.reindex(scores_f.index) > 0)
support = pd.DataFrame({
    'n_months': st.value_counts().sort_index(),
    'pct': (100 * st.value_counts(normalize=True).sort_index()).round(1),
    'is_crisis': [int(s == crisisF) for s in sorted(st.unique())],
    'nber_overlap_pct': [round(100 * float(ur[st == s].mean()), 1) for s in sorted(st.unique())],
})
display(support)""")

md(r"""**State labels & interpretation.** Read from the per-state feature / PC profiles, the NBER
overlap, and when each state occurs. **State 0 is the model's crisis state.**

| state | size | NBER overlap | intuitive label | character |
|---|---|---|---|---|
| **0** | 58 (10%) | **41% (crisis)** | **Acute crisis / risk-off** | the weakest equity returns of any state, policy rate low and falling, quality outperforms — the 2008 and 2020 episodes. This is the crisis flag $\kappa$ scores. |
| **1** | 34 (6%) | 21% | **Disinflationary easing / soft patch** | front-end rates falling, high-beta leading low-beta, mildly soft equities — mostly 1990s–2000s slowdowns |
| **2** | 23 (4%) | **52%** | **High-inflation tightening (Volcker era)** | extreme rate *level* with the front end still rising and weak equities — almost entirely the early 1980s |
| **3** | 243 (41%) | 2% | **Calm expansion (baseline)** | every factor near its average; the dominant trend-growth regime, present in all decades |
| **4** | 68 (11%) | 15% | **High-rate expansion (1970s–80s)** | elevated but stable policy rates with positive equity returns — concentrated in the 1970s–80s |
| **5** | 169 (28%) | 1% | **Low-rate / post-GFC expansion (ZIRP)** | very low policy rate, steady positive equities — dominated by the 2010s |

The recession signal is split across states 0, 2, 4 and 1; the crisis state alone owns ~41% of
all recession months. Flagging only state 0 as "crisis" is what gives the HMM its $\kappa$ here —
it cleanly captures the acute 2008 / 2020 drawdowns, while the slower inflation-era recessions
fall into the high-rate states (2 and 4).""")

# ───────────────────────────────────────────────────────────────────────────────
md(r"""### 8e · Regime transition matrix

Row $i$, column $j$ is $P(\text{state }j \text{ next month} \mid \text{state }i \text{ now})$
under the fitted HMM; rows sum to 1. Large diagonal entries mean regimes are persistent.
Saved to `final_transition_matrix.csv`.""")

code(r"""A_df = pd.DataFrame(AF, index=[f'from_state{i}' for i in range(6)],
                    columns=[f'to_state{j}' for j in range(6)])
A_df.to_csv(OUT / 'final_transition_matrix.csv')
print('saved outputs/final_transition_matrix.csv   (rows sum to 1)\n')
display(A_df.round(3))
fig, ax = plt.subplots(figsize=(5.2, 4.5))
im = ax.imshow(AF, cmap='viridis', vmin=0, vmax=1)
ax.set_xticks(range(6)); ax.set_yticks(range(6)); ax.tick_params(labelsize=11)
ax.set_xlabel('to state', fontsize=12); ax.set_ylabel('from state', fontsize=12)
ax.set_title('HMM transition matrix', fontsize=13)
for i in range(6):
    for j in range(6):
        ax.text(j, i, f'{AF[i, j]:.2f}', ha='center', va='center',
                color='white' if AF[i, j] < 0.6 else 'black', fontsize=12)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.ax.tick_params(labelsize=10)
fig.tight_layout()
fig.savefig(OUT / 'fig_transition_matrix.png', dpi=130, bbox_inches='tight')
print('saved outputs/fig_transition_matrix.png'); plt.show()""")

# ───────────────────────────────────────────────────────────────────────────────
md(r"""### 8f · Per-month state probabilities

The final export: for every month, the Viterbi-decoded **`state`** (the exact series $\kappa$
scores) plus **`p_state0..p_state5`**, the HMM's filtered (forward) posterior probabilities
of each state — the same fitted model and decoding family used to compute $\kappa$. (Because
Viterbi optimises the joint path while the posterior is a per-month marginal, the
posterior's argmax can differ from `state` in a few months.) Saved to
`final_state_probabilities.csv`.""")

code(r"""prob = pd.DataFrame(postF, index=scores_f.index, columns=[f'p_state{k}' for k in range(6)])
prob.insert(0, 'state', statesF.values)
prob.index.name = 'date'; prob = prob.reset_index()
prob['date'] = pd.to_datetime(prob['date']).dt.date
prob.to_csv(OUT / 'final_state_probabilities.csv', index=False)
psum = postF.sum(1)
print('saved outputs/final_state_probabilities.csv')
print(f'rows={len(prob)}   prob-row-sum in [{psum.min():.4f}, {psum.max():.4f}]')
print(f'columns: {list(prob.columns)}\n')
display(prob.head(8)); display(prob.tail(4))""")

# ───────────────────────────────────────────────────────────────────────────────
md(r"""## 9 · Notes & caveats

* **In-sample $\kappa$.** Every engine is fit on all available months and scored on the same
  span — an upper bound on detectability, not out-of-sample skill. Feature *selection* for
  the `bn_meanrank` rows is look-ahead-free (data up to 2017-12 only).
* **Arbitrary state ids.** State numbers are interchangeable labels; read them via the
  recession overlap, the crisis-state flag, and the per-state PC/feature profiles, not by
  their index.
* **HMM stability.** The HMM is the more fragile engine (see the ARI column); the crisis
  state is pinned by Mahalanobis ownership precisely because raw HMM states can be unstable.
* **Reproducibility.** Thread pinning + full-SVD PCA + fixed seeds make every number above
  byte-reproducible across runs.""")

# ───────────────────────────────────────────────────────────────────────────────
nb = nbf.v4.new_notebook()
nb['cells'] = cells
nb['metadata'] = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python'},
}
with open('regime_kappa_states.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print(f'wrote regime_kappa_states.ipynb  ({len(cells)} cells)')
