"""
regime_taa.py — Core library for replicating

    "Tactical Asset Allocation with Macroeconomic Regime Detection"
    Oliveira, Sandfelder, Fujita, Dong, Cucuringu (arXiv:2503.11499v2).

All heavy / reused logic lives here so the two notebooks
(``01_regime_detection.ipynb`` and ``02_tactical_allocation.ipynb``) stay
readable.  Equation numbers in docstrings refer to the paper.

Three stages:
  Stage 1  Regime classification        -> RegimeModel (Algorithm 1)
  Stage 2  Probabilities + transitions  -> Eq 1-5
  Stage 3  Forecasting + allocation     -> Eq 6-19, metrics
"""
from __future__ import annotations

import io
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

try:  # optional, only used by elbow_k()
    from kneed import KneeLocator
except Exception:  # pragma: no cover
    KneeLocator = None


# ===========================================================================
# Constants
# ===========================================================================

#: International exchange-rate series within FRED-MD's "Interest and Exchange
#: Rates" group.  The paper "excluded group 6 to focus only on U.S.
#: macroeconomic data" -- exchange rates are the *non-U.S.* part of that group,
#: so we drop these by default while KEEPING U.S. interest rates (which the
#: paper's Table 1 / Fig 4 use to characterise regimes, e.g. FEDFUNDS).
EXCHANGE_RATE_SERIES = [
    "TWEXAFEGSMTHx", "TWEXMMTH", "EXSZUSx", "EXJPUSx", "EXUSUKx", "EXCAUSx",
]

#: Full McCracken & Ng group 6 (interest + exchange rates) -- the stricter,
#: literal reading of "excluded group 6". Selectable in prepare_features().
INTEREST_EXCHANGE_SERIES = [
    "FEDFUNDS", "CP3Mx", "TB3MS", "TB6MS", "GS1", "GS5", "GS10", "AAA", "BAA",
    "COMPAPFFx", "TB3SMFFM", "TB6SMFFM", "T1YFFM", "T5YFFM", "T10YFFM",
    "AAAFFM", "BAAFFM",
] + EXCHANGE_RATE_SERIES

#: Descriptive series used in Fig 4 (raw levels, averaged per regime).
FIG4_SERIES = ["RPI", "UNRATE", "UMCSENTx", "FEDFUNDS", "CPIAUCSL", "S&P 500"]

#: The 10 ETFs (Table 2).
ETF_TICKERS = ["SPY", "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]

FREDMD_URL = ("https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/"
              "research/fred-md/monthly/current.csv")
USREC_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=USREC"

MONTHS_PER_YEAR = 12


# ===========================================================================
# FRED-MD loading & transformation
# ===========================================================================

def download_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8", errors="replace")


def load_fredmd(path_or_url: str = FREDMD_URL):
    """Return (data, tcodes) where ``data`` is a DataFrame indexed by month-start
    timestamps (raw levels) and ``tcodes`` is a Series of transform codes."""
    if path_or_url.startswith("http"):
        raw = download_text(path_or_url)
    else:
        with open(path_or_url, "r", encoding="utf-8", errors="replace") as fh:
            raw = fh.read()
    df = pd.read_csv(io.StringIO(raw))
    df = df.rename(columns={df.columns[0]: "sasdate"})
    # Row 0 holds the per-column transform codes ("Transform:").
    tcode_row = df.iloc[0]
    tcodes = tcode_row.drop(labels=["sasdate"]).astype(float).astype(int)
    data = df.iloc[1:].copy()
    data = data[data["sasdate"].notna()]
    data["sasdate"] = pd.to_datetime(data["sasdate"], errors="coerce")
    data = data[data["sasdate"].notna()].set_index("sasdate").sort_index()
    data = data.apply(pd.to_numeric, errors="coerce")
    # Normalise index to month-start (paper: "first day of each month").
    data.index = data.index.to_period("M").to_timestamp()
    return data, tcodes


def _transform_series(x: pd.Series, tcode: int) -> pd.Series:
    """Apply one FRED-MD transform code (1-7)."""
    if tcode == 1:                       # level
        return x
    if tcode == 2:                       # first difference
        return x.diff()
    if tcode == 3:                       # second difference
        return x.diff().diff()
    if tcode == 4:                       # log
        return np.log(x)
    if tcode == 5:                       # first difference of log
        return np.log(x).diff()
    if tcode == 6:                       # second difference of log
        return np.log(x).diff().diff()
    if tcode == 7:                       # delta of pct change
        return (x / x.shift(1) - 1.0).diff()
    raise ValueError(f"unknown tcode {tcode!r}")


def apply_transforms(data: pd.DataFrame, tcodes: pd.Series) -> pd.DataFrame:
    out = {}
    for col in data.columns:
        if col in tcodes.index:
            out[col] = _transform_series(data[col], int(tcodes[col]))
    return pd.DataFrame(out, index=data.index)


def load_usrec(path_or_url: str = USREC_URL) -> pd.Series:
    """NBER recession indicator (monthly, 0/1) for figure shading."""
    if path_or_url.startswith("http"):
        raw = download_text(path_or_url)
        df = pd.read_csv(io.StringIO(raw))
    else:
        df = pd.read_csv(path_or_url)
    df.columns = [c.lower() for c in df.columns]
    date_col = [c for c in df.columns if "date" in c][0]
    val_col = [c for c in df.columns if c != date_col][0]
    s = pd.Series(
        pd.to_numeric(df[val_col], errors="coerce").values,
        index=pd.to_datetime(df[date_col]).dt.to_period("M").dt.to_timestamp(),
        name="USREC",
    )
    return s.dropna()


# ===========================================================================
# Feature preparation: transform -> clean -> standardise -> PCA
# ===========================================================================

@dataclass
class FeatureBundle:
    scores: pd.DataFrame          # PCA scores, index = months
    transformed: pd.DataFrame     # cleaned transformed features (pre-PCA)
    scaler: StandardScaler
    pca: PCA
    n_components: int
    columns: list                 # feature columns fed to PCA


def remove_outliers_iqr(feats: pd.DataFrame, k: float = 10.0) -> pd.DataFrame:
    """Standard McCracken-Ng FRED-MD outlier rule: values with
    |x - median| > k * IQR are set to NaN (later imputed).

    Without this, extreme shocks (e.g. the 2020 COVID collapse in
    second-differenced series) dominate the PCA and make the l2 outlier split
    degenerate (a single month)."""
    med = feats.median()
    iqr = feats.quantile(0.75) - feats.quantile(0.25)
    mask = (feats - med).abs() > (k * iqr)
    return feats.mask(mask)


def prepare_features(
    data: pd.DataFrame,
    tcodes: pd.Series,
    exclude: str = "exchange",
    pca_var: float = 0.95,
    drop_initial: int = 2,
    remove_outliers: bool = True,
):
    """Full Section 4.2 pipeline.

    ``exclude`` selects which series to drop before clustering:
      "exchange" (default) -> international FX series only (keeps U.S. rates);
      "group6"             -> the full interest+exchange group (literal reading);
      "none"               -> keep everything.
    ``drop_initial`` removes the first rows lost to second-differencing.
    ``remove_outliers`` applies the standard 10*IQR rule before imputing
    remaining gaps (interpolation + median fill).
    """
    feats = apply_transforms(data, tcodes)
    drop_map = {"exchange": EXCHANGE_RATE_SERIES,
                "group6": INTEREST_EXCHANGE_SERIES, "none": []}
    drop = [c for c in drop_map[exclude] if c in feats.columns]
    feats = feats.drop(columns=drop)
    feats = feats.iloc[drop_initial:]
    if remove_outliers:
        feats = remove_outliers_iqr(feats)
    # Drop sparse columns, impute remaining gaps so PCA sees a full matrix.
    feats = feats.dropna(axis=1, thresh=int(0.95 * len(feats)))
    feats = feats.interpolate(limit_direction="both").fillna(feats.median())
    feats = feats.dropna(axis=0)

    scaler = StandardScaler()
    z = scaler.fit_transform(feats.values)

    pca_full = PCA().fit(z)
    cum = np.cumsum(pca_full.explained_variance_ratio_)
    n_comp = int(np.searchsorted(cum, pca_var) + 1)
    pca = PCA(n_components=n_comp).fit(z)
    scores = pd.DataFrame(
        pca.transform(z),
        index=feats.index,
        columns=[f"PC{i+1}" for i in range(n_comp)],
    )
    return FeatureBundle(scores, feats, scaler, pca, n_comp, list(feats.columns))


def pca_explained_variance(z: np.ndarray) -> np.ndarray:
    """Cumulative explained-variance ratio for Fig 1."""
    return np.cumsum(PCA().fit(z).explained_variance_ratio_)


# ===========================================================================
# Stage 1: clustering primitives
# ===========================================================================

def _l2_distances(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Euclidean distance from each row of X to each centroid -> (n, k)."""
    return np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)


def _cosine_distances(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Cosine distance (1 - cosine similarity) -> (n, k)."""
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    Cn = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)
    return 1.0 - Xn @ Cn.T


def membership_probs(distances: np.ndarray) -> np.ndarray:
    """Eq 1: P(C_i) = (1 - d_i/sum_j d_j) / sum_m(1 - d_m/sum_j d_j).

    ``distances`` is (n, k); returns (n, k) rows summing to 1.
    """
    d = np.asarray(distances, dtype=float)
    tot = d.sum(axis=1, keepdims=True) + 1e-12
    num = 1.0 - d / tot
    return num / (num.sum(axis=1, keepdims=True) + 1e-12)


def elbow_k(X: np.ndarray, k_min: int = 2, k_max: int = 10,
            normalize: bool = True, random_state: int = 0):
    """Pick the elbow value of k for (spherical) k-means via kneed.

    Returns (best_k, ks, inertias).  Falls back to max-curvature if kneed is
    unavailable.
    """
    Xf = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12) if normalize else X
    ks = list(range(k_min, k_max + 1))
    inertias = [
        KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(Xf).inertia_
        for k in ks
    ]
    best = None
    if KneeLocator is not None:
        kn = KneeLocator(ks, inertias, curve="convex", direction="decreasing")
        best = kn.knee
    if best is None:  # fallback: largest second difference
        d2 = np.diff(inertias, 2)
        best = ks[int(np.argmax(d2)) + 1] if len(d2) else ks[0]
    return int(best), ks, inertias


# ===========================================================================
# Stage 1+2: the RegimeModel (Algorithm 1 + Eq 1-4)
# ===========================================================================

@dataclass
class RegimeModel:
    """Layered regime detector with probabilistic memberships.

    Fit on PCA scores X (n, p).  ``r`` is the number of cosine regimes
    (regimes 1..r); regime 0 is the outlier/economic-difficulty regime.

    ``outlier_method`` selects how Regime 0 is identified:
      "kmeans"   -- Algorithm 1 literal: l2 k-means (k=2), smaller cluster =
                    Regime 0. Faithful to the paper, but on the current data the
                    split is bimodal (a single COVID month without outlier
                    removal, or a balanced ~37% split with it).
      "quantile" -- flag the ``outlier_frac`` most statistically extreme months
                    (Mahalanobis distance to the global centroid) as Regime 0.
                    Guarantees a *sparse, crisis-focused* Regime 0 like the
                    paper's, and keeps the soft-membership/Eq 4 machinery via a
                    logistic in the distance. This is the documented "fix" for
                    the over-prevalent Regime 0; it deviates from the literal
                    Algorithm 1.
    The cosine k-means on the typical months (regimes 1..r) is identical either
    way.
    """
    r: int = 5
    random_state: int = 0
    n_init: int = 10
    outlier_method: str = "kmeans"
    outlier_frac: float = 0.15
    # learned (kmeans)
    l2_centroids: np.ndarray = field(default=None, repr=False)
    outlier_idx: int = 0
    # learned (quantile)
    center_: np.ndarray = field(default=None, repr=False)
    var_: np.ndarray = field(default=None, repr=False)
    threshold_: float = field(default=None, repr=False)
    dist_scale_: float = field(default=None, repr=False)
    # shared
    cos_centroids: np.ndarray = field(default=None, repr=False)  # (r, p) raw space
    labels_: np.ndarray = field(default=None, repr=False)

    @property
    def n_regimes(self) -> int:
        return self.r + 1

    def _maha(self, X: np.ndarray) -> np.ndarray:
        """Mahalanobis distance to the fitted centre (diagonal cov = PC vars)."""
        Z = np.asarray(X, dtype=float) - self.center_
        return np.sqrt((Z * Z / self.var_).sum(axis=1))

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        # --- identify the outlier (Regime 0) months -------------------------
        if self.outlier_method == "quantile":
            self.center_ = X.mean(axis=0)
            self.var_ = X.var(axis=0, ddof=1) + 1e-12
            d = self._maha(X)
            self.threshold_ = float(np.quantile(d, 1.0 - self.outlier_frac))
            self.dist_scale_ = float(np.median(np.abs(d - np.median(d)))) + 1e-9
            typical_mask = d < self.threshold_
        else:  # "kmeans" (Algorithm 1)
            km2 = KMeans(n_clusters=2, n_init=self.n_init,
                         random_state=self.random_state).fit(X)
            lab2 = km2.labels_
            self.outlier_idx = int(np.argmin(np.bincount(lab2, minlength=2)))
            self.l2_centroids = km2.cluster_centers_
            typical_mask = lab2 != self.outlier_idx
        B = X[typical_mask]
        # --- cosine k-means on typical months (spherical k-means) ----------
        Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
        kmc = KMeans(n_clusters=self.r, n_init=self.n_init,
                     random_state=self.random_state).fit(Bn)
        self.cos_centroids = np.vstack([
            B[kmc.labels_ == j].mean(axis=0) for j in range(self.r)
        ])
        self.labels_ = self.hard_labels(X)
        return self

    def _is_outlier(self, X: np.ndarray) -> np.ndarray:
        if self.outlier_method == "quantile":
            return self._maha(X) >= self.threshold_
        d2 = _l2_distances(X, self.l2_centroids)
        return np.argmin(d2, axis=1) == self.outlier_idx

    def _p_regime0(self, X: np.ndarray) -> np.ndarray:
        if self.outlier_method == "quantile":
            z = (self._maha(X) - self.threshold_) / self.dist_scale_
            return 1.0 / (1.0 + np.exp(-z))          # logistic, 0.5 at threshold
        p_l2 = membership_probs(_l2_distances(X, self.l2_centroids))
        return p_l2[:, self.outlier_idx]

    # ---- hard assignment ----------------------------------------------------
    def hard_labels(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        is_outlier = self._is_outlier(X)
        cos_lab = np.argmin(_cosine_distances(X, self.cos_centroids), axis=1) + 1
        return np.where(is_outlier, 0, cos_lab).astype(int)

    # ---- probabilistic distribution over {0..r} (Eq 1-4) -------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        p_r0 = self._p_regime0(X)                              # P(Regime 0)
        p_cos = membership_probs(_cosine_distances(X, self.cos_centroids))  # (n, r)
        p_max = p_cos.max(axis=1)                              # Eq 2
        # Eq 4: P_R0 = -P_max * log2(1 - P(Regime 0)).
        p_r0c = np.clip(p_r0, 0.0, 1.0 - 1e-9)
        scaled_r0 = -p_max * np.log2(1.0 - p_r0c)              # (n,)
        full = np.column_stack([scaled_r0, p_cos])             # (n, r+1)
        full = full / (full.sum(axis=1, keepdims=True) + 1e-12)
        return full


# ===========================================================================
# Stage 2: transition matrices (Eq 5) + centroid matching
# ===========================================================================

def transition_matrix(labels: np.ndarray, n_regimes: int) -> np.ndarray:
    """Eq 5: E[i, j] = count(i -> j) / |Regime i|  (row-stochastic).

    Row i = P(next regime = j | current regime = i).  Note Fig 5 displays the
    transpose ("To" on rows).
    """
    labels = np.asarray(labels, dtype=int)
    E = np.zeros((n_regimes, n_regimes))
    for cur, nxt in zip(labels[:-1], labels[1:]):
        E[cur, nxt] += 1
    counts = E.sum(axis=1, keepdims=True)
    counts[counts == 0] = 1.0
    return E / counts


def transition_given_switch(E: np.ndarray) -> np.ndarray:
    """Off-diagonals of row i divided by (1 - diag(i)); diagonal set to 0.

    'Transition probabilities given that a transition takes place' (Section 4.6).
    """
    n = E.shape[0]
    M = E.copy()
    diag = np.diag(E).copy()
    for i in range(n):
        denom = 1.0 - diag[i]
        if denom > 1e-12:
            M[i] = M[i] / denom
        M[i, i] = 0.0
    return M


def match_centroids(new_centroids: np.ndarray,
                    ref_centroids: np.ndarray) -> np.ndarray:
    """Hungarian matching of new cosine centroids to a reference set.

    Returns a permutation ``perm`` such that ``new_centroids[perm]`` aligns
    with ``ref_centroids`` (maximising cosine similarity).  Used to keep regime
    labels consistent across walk-forward re-fits ("matching clustering
    algorithm to ensure consistency").
    """
    A = new_centroids / (np.linalg.norm(new_centroids, axis=1, keepdims=True) + 1e-12)
    B = ref_centroids / (np.linalg.norm(ref_centroids, axis=1, keepdims=True) + 1e-12)
    sim = A @ B.T                       # (k, k) cosine similarity
    row, col = linear_sum_assignment(-sim)
    perm = np.empty(len(row), dtype=int)
    perm[col] = row                     # ref slot col <- new cluster row
    return perm


# ===========================================================================
# Stage 3: regime forecasting (Eq 6-7)
# ===========================================================================

def forecast_regime_dist(p_t: np.ndarray, E: np.ndarray) -> np.ndarray:
    """Eq 6-7: normalise p_t then p_bar_{t+1} = p_bar_t^T E."""
    p = np.asarray(p_t, dtype=float)
    p = p / (p.sum() + 1e-12)
    nxt = p @ E
    return nxt / (nxt.sum() + 1e-12)


# ===========================================================================
# Stage 3: forecasting models -> per-asset score vectors
# ===========================================================================

def conditional_stats(returns: pd.DataFrame, regime_labels: np.ndarray,
                      target_regime: int):
    """Sample mean & std of each asset's returns over months in target_regime.

    ``returns`` (T, d), ``regime_labels`` (T,) aligned by row.
    Returns (mu, sigma) length-d Series (NaN-safe: falls back to full sample).
    """
    mask = regime_labels == target_regime
    if mask.sum() < 2:
        sub = returns
    else:
        sub = returns.iloc[mask]
    return sub.mean(), sub.std(ddof=1)


def naive_scores(returns: pd.DataFrame, regime_labels: np.ndarray,
                 next_regime: int) -> np.ndarray:
    """Eq 8-10: conditional Sharpe ratio per ETF for the predicted regime."""
    mu, sigma = conditional_stats(returns, regime_labels, next_regime)
    s = mu / sigma.replace(0, np.nan)
    return s.fillna(0.0).values


def bl_view_means(returns: pd.DataFrame, regime_labels: np.ndarray,
                  next_regime: int) -> np.ndarray:
    """Eq 11: regime-conditional mean returns used as BL views q*."""
    mu, _ = conditional_stats(returns, regime_labels, next_regime)
    return mu.fillna(0.0).values


def ridge_scores(features: pd.DataFrame, returns: pd.DataFrame,
                 regime_labels: np.ndarray, regime_dist: np.ndarray,
                 latest_features: np.ndarray, n_regimes: int,
                 alpha: float = 1.0) -> np.ndarray:
    """Eq 12-14: per-regime ridge of returns on macro features, aggregated by
    the forecast regime distribution.

    ``features`` (T, p) aligned with ``returns`` (T, d) and ``regime_labels``.
    ``latest_features`` (p,) is the most recent macro vector.
    """
    d = returns.shape[1]
    agg = np.zeros(d)
    x_pred = latest_features.reshape(1, -1)
    for i in range(n_regimes):
        w_i = regime_dist[i]
        if w_i <= 1e-6:
            continue
        mask = regime_labels == i
        if mask.sum() < max(5, features.shape[1] // 4):
            # too little data in this regime -> use unconditional mean
            pred_i = returns.mean().values
        else:
            Xi = features.iloc[mask].values
            Yi = returns.iloc[mask].values
            model = Ridge(alpha=alpha)
            model.fit(Xi, Yi)
            pred_i = model.predict(x_pred)[0]
        agg += w_i * pred_i
    return agg


def mvo_scores(returns: pd.DataFrame, ridge_shrink: float = 1e-4) -> np.ndarray:
    """Unconstrained mean-variance score ~ Sigma^{-1} mu (control model)."""
    mu = returns.mean().values
    cov = returns.cov().values
    cov = cov + ridge_shrink * np.eye(cov.shape[0])
    return np.linalg.solve(cov, mu)


def bl_scores(returns: pd.DataFrame, views: np.ndarray,
              tau: float = 0.05, view_conf: float = 1.0,
              ridge_shrink: float = 1e-4) -> np.ndarray:
    """Eq 19: Black-Litterman posterior mean used as the BL score vector.

    Views are absolute (P = I).  Omega = view_conf * diag(tau * Sigma).
    """
    mu = returns.mean().values
    Sigma = returns.cov().values + ridge_shrink * np.eye(returns.shape[1])
    n = len(mu)
    P = np.eye(n)
    tauS_inv = np.linalg.inv(tau * Sigma)
    Omega = view_conf * np.diag(np.diag(tau * Sigma))
    Omega_inv = np.linalg.inv(Omega + ridge_shrink * np.eye(n))
    A = tauS_inv + P.T @ Omega_inv @ P
    b = tauS_inv @ mu + P.T @ Omega_inv @ views
    return np.linalg.solve(A, b)


# ===========================================================================
# Stage 3: position sizing (Eq 15-18)
# ===========================================================================

def apply_sizing(scores: np.ndarray, style: str, l: int,
                 next_regime: int | None = None) -> np.ndarray:
    """Map a per-asset score vector to portfolio weights in [-1, 1]^d.

    style: 'lo' (long-only, Eq 18), 'lns' (long-short, Eq 16),
           'los' (long-or-short largest magnitude, Eq 17),
           'mx'  (mixed: lns if next_regime == 0 else lo).
    """
    s = np.asarray(scores, dtype=float)
    d = len(s)
    l = min(l, d)
    w = np.zeros(d)

    if style == "mx":
        style = "lns" if next_regime == 0 else "lo"

    if style == "lns":
        order = np.argsort(s)
        idx = np.concatenate([order[:l], order[-l:]])          # l lowest + l highest
        idx = np.unique(idx)
        denom = np.abs(s[idx]).sum()
        if denom > 1e-12:
            w[idx] = s[idx] / denom
    elif style == "los":
        idx = np.argsort(np.abs(s))[-l:]                       # largest magnitude
        denom = np.abs(s[idx]).sum()
        if denom > 1e-12:
            w[idx] = s[idx] / denom
    elif style == "lo":
        idx = np.argsort(s)[-l:]                               # l highest
        pos = np.clip(s[idx], 0.0, None)                       # long-only: drop negatives
        denom = pos.sum()
        if denom > 1e-12:
            w[idx] = pos / denom
    else:
        raise ValueError(f"unknown style {style!r}")
    return w


# ===========================================================================
# Stage 3: volatility scaling
# ===========================================================================

def vol_scale_weights(weights: np.ndarray, cov_monthly: np.ndarray,
                      target_annual: float = 0.10,
                      max_leverage: float = 5.0) -> np.ndarray:
    """Scale weights so annualised portfolio vol = target (Figs 10-13)."""
    w = np.asarray(weights, dtype=float)
    var_m = float(w @ cov_monthly @ w)
    if var_m <= 1e-12:
        return w
    vol_annual = np.sqrt(var_m) * np.sqrt(MONTHS_PER_YEAR)
    factor = target_annual / vol_annual
    factor = min(factor, max_leverage)
    return w * factor


# ===========================================================================
# Stage 3: performance metrics
# ===========================================================================

def _to_series(r) -> pd.Series:
    return r if isinstance(r, pd.Series) else pd.Series(np.asarray(r, dtype=float))


def sharpe_ratio(returns, periods: int = MONTHS_PER_YEAR) -> float:
    r = _to_series(returns).dropna()
    if r.std(ddof=1) == 0 or len(r) < 2:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * np.sqrt(periods))


def sortino_ratio(returns, periods: int = MONTHS_PER_YEAR) -> float:
    r = _to_series(returns).dropna()
    downside = r[r < 0]
    dd = downside.std(ddof=1) if len(downside) > 1 else np.nan
    if not dd or np.isnan(dd) or dd == 0:
        return 0.0
    return float(r.mean() / dd * np.sqrt(periods))


def drawdown_series(returns) -> pd.Series:
    r = _to_series(returns).fillna(0.0)
    cum = (1.0 + r).cumprod()
    peak = cum.cummax()
    return (cum / peak - 1.0)


def max_drawdown(returns) -> float:
    """Maximum drawdown, as a percentage (negative)."""
    return float(drawdown_series(returns).min() * 100.0)


def avg_drawdown(returns) -> float:
    return float(drawdown_series(returns).mean() * 100.0)


def pct_positive(returns) -> float:
    r = _to_series(returns).dropna()
    return float((r > 0).mean()) if len(r) else 0.0


def performance_table(returns_dict: dict) -> pd.DataFrame:
    """Build a Sharpe/Sortino/MaxDD/%Positive table (Tables 4-6)."""
    rows = {}
    for name, r in returns_dict.items():
        rows[name] = {
            "Sharpe": round(sharpe_ratio(r), 3),
            "Sortino": round(sortino_ratio(r), 3),
            "MaxDD": round(max_drawdown(r), 3),
            "% Positive Ret.": round(pct_positive(r), 3),
        }
    return pd.DataFrame(rows).T


# ===========================================================================
# Stage 3: walk-forward backtest harness
# ===========================================================================

#: Sizing styles used per model (matches the variants tabulated in the paper).
#: naive & ridge use all four styles (Tables 4, 6); BL & MVO use lns/lo (Table 5).
MODEL_STYLES = {
    "naive": ("lns", "mx", "los", "lo"),
    "ridge": ("lns", "mx", "los", "lo"),
    "bl": ("lns", "lo"),
    "mvo": ("lns", "lo"),
}


def run_backtest(
    features: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    window: int = 48,
    r: int = 5,
    randomize_regimes: bool = False,
    seed: int = 0,
    ridge_alpha: float = 10.0,
    ridge_n_features: int = 10,
    vol_target: float = 0.10,
    ls=(2, 3, 4),
    models=("naive", "ridge", "bl", "mvo"),
    kmeans_n_init: int = 3,
    outlier_method: str = "kmeans",
    outlier_frac: float = 0.15,
    include_benchmarks: bool = True,
) -> pd.DataFrame:
    """Fixed-window walk-forward backtest (Sections 5-6).

    At each month t the model is fit / forecast on the trailing ``window`` of
    months, weights are formed for every (model, style, l) variant, scaled to
    ``vol_target`` annualised vol, and applied to month t's realised returns.

    ``features`` (T, p) PCA scores and ``returns`` (T, d) ETF returns must share
    the same monthly index.  When ``randomize_regimes`` is True the regime
    labels/distribution are drawn at random (the paper's control), destroying
    the regime signal while keeping the rest of the pipeline identical.

    Returns a DataFrame of realised monthly returns, one column per strategy
    (plus ``spy`` / ``ew`` benchmarks), indexed by realisation month.
    """
    feats = features.values
    idx = returns.index
    T, d = returns.shape
    n_reg = r + 1
    rng = np.random.default_rng(seed)
    cols = list(returns.columns)
    spy_col = cols.index("SPY") if "SPY" in cols else 0

    records = defaultdict(list)
    rec_idx = []

    for t in range(window, T):
        Xtr = feats[t - window:t]
        Rtr = returns.iloc[t - window:t]
        realized = returns.iloc[t].values

        if randomize_regimes:
            labels = rng.integers(0, n_reg, size=window)
            p_t = rng.dirichlet(np.ones(n_reg))
        else:
            model = RegimeModel(r=r, random_state=0, n_init=kmeans_n_init,
                                outlier_method=outlier_method,
                                outlier_frac=outlier_frac).fit(Xtr)
            labels = model.labels_
            p_t = model.predict_proba(Xtr[-1:])[0]

        E = transition_matrix(labels, n_reg)
        p_next = forecast_regime_dist(p_t, E)
        istar = int(np.argmax(p_next))
        cov = Rtr.cov().values

        scores = {}
        if "naive" in models:
            scores["naive"] = naive_scores(Rtr, labels, istar)
        if "ridge" in models:
            k = min(ridge_n_features, Xtr.shape[1])
            Xr = pd.DataFrame(Xtr[:, :k], index=Rtr.index)
            scores["ridge"] = ridge_scores(
                Xr, Rtr, labels, p_next, Xtr[-1, :k], n_reg, alpha=ridge_alpha)
        if "mvo" in models:
            scores["mvo"] = mvo_scores(Rtr)
        if "bl" in models:
            scores["bl"] = bl_scores(Rtr, bl_view_means(Rtr, labels, istar))

        for mname, sc in scores.items():
            for style in MODEL_STYLES[mname]:
                for l in ls:
                    w = apply_sizing(sc, style, l, next_regime=istar)
                    w = vol_scale_weights(w, cov, vol_target)
                    records[f"{mname}_{style}_{l}"].append(float(w @ realized))

        if include_benchmarks:
            w_ew = np.ones(d) / d
            records["ew"].append(float(vol_scale_weights(w_ew, cov, vol_target) @ realized))
            w_spy = np.zeros(d)
            w_spy[spy_col] = 1.0
            records["spy"].append(float(vol_scale_weights(w_spy, cov, vol_target) @ realized))

        rec_idx.append(idx[t])

    return pd.DataFrame(records, index=pd.DatetimeIndex(rec_idx))


def walk_forward_regimes(
    features: pd.DataFrame,
    *,
    window: int = 48,
    expanding: bool = False,
    min_train: int = 48,
    r: int = 5,
    outlier_method: str = "quantile",
    outlier_frac: float = 0.15,
    kmeans_n_init: int = 3,
    reference: "RegimeModel | None" = None,
) -> pd.Series:
    """Live (walk-forward) regime classification with cross-window label
    consistency.

    For each month m, fit a RegimeModel on the data up to m and record the
    regime assigned to month m using only data up to m -- exactly what the
    backtest sees in real time. With ``expanding=False`` the fit uses the
    trailing ``window`` months; with ``expanding=True`` it uses ALL history up
    to m (starting once ``min_train`` months are available). Each fit's cosine
    labels are arbitrary, so they are Hungarian-matched to ``reference`` (a
    full-sample fit by default) for *consistent labels/colours only* -- this
    alignment affects no allocation decision.

    Returns a Series of consistent regime labels indexed by month.
    """
    X = np.asarray(features.values, dtype=float)
    idx = features.index
    T = len(X)
    if reference is None:
        reference = RegimeModel(r=r, outlier_method=outlier_method,
                                outlier_frac=outlier_frac).fit(X)
    ref_cos = reference.cos_centroids
    out = {}
    start = (min_train - 1) if expanding else (window - 1)
    for m in range(start, T):
        win = X[: m + 1] if expanding else X[m - window + 1: m + 1]
        mdl = RegimeModel(r=r, n_init=kmeans_n_init, outlier_method=outlier_method,
                          outlier_frac=outlier_frac).fit(win)
        lbl = int(mdl.hard_labels(X[m:m + 1])[0])
        if lbl != 0:  # align cosine cluster (lbl-1) to the reference numbering
            A = mdl.cos_centroids / (np.linalg.norm(mdl.cos_centroids, axis=1,
                                                    keepdims=True) + 1e-12)
            B = ref_cos / (np.linalg.norm(ref_cos, axis=1, keepdims=True) + 1e-12)
            row, col = linear_sum_assignment(-(A @ B.T))
            new_to_ref = np.empty(r, dtype=int)
            new_to_ref[row] = col
            lbl = int(new_to_ref[lbl - 1]) + 1
        out[idx[m]] = lbl
    return pd.Series(out, name="wf_regime")


def run_backtest_expanding(
    features: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    min_train: int = 48,
    r: int = 5,
    outlier_method: str = "quantile",
    outlier_frac: float = 0.15,
    randomize_regimes: bool = False,
    seed: int = 0,
    ridge_alpha: float = 10.0,
    ridge_n_features: int = 10,
    vol_target: float = 0.10,
    ls=(2, 3, 4),
    models=("naive", "ridge", "bl", "mvo"),
    kmeans_n_init: int = 3,
    include_benchmarks: bool = True,
) -> pd.DataFrame:
    """Expanding-window walk-forward backtest.

    Unlike :func:`run_backtest`'s fixed trailing window, the regime model is
    re-fit on ALL macro history up to each step. ``features`` may start decades
    before ``returns`` (e.g. FRED-MD from 1959 while the ETFs start in 2000):
    the regime model uses the long macro history, while the asset statistics use
    the available return history (the overlap of ``returns`` with the regime
    labels). Realisation begins after ``min_train`` return months.

    Returns realised monthly returns per strategy, indexed by realisation month.
    """
    n_reg = r + 1
    d = returns.shape[1]
    cols = list(returns.columns)
    spy_col = cols.index("SPY") if "SPY" in cols else 0
    rng = np.random.default_rng(seed)
    ret_idx = returns.index
    records = defaultdict(list)
    rec_idx = []

    for k in range(min_train, len(ret_idx)):
        m = ret_idx[k]
        feat_tr = features.loc[features.index < m]      # all macro history < m
        rets_tr = returns.iloc[:k]                      # ETF returns < m
        realized = returns.iloc[k].values

        if randomize_regimes:
            ret_labels = rng.integers(0, n_reg, size=len(rets_tr))
            p_t = rng.dirichlet(np.ones(n_reg))
            E = transition_matrix(rng.integers(0, n_reg, size=len(feat_tr)), n_reg)
        else:
            mdl = RegimeModel(r=r, n_init=kmeans_n_init, outlier_method=outlier_method,
                              outlier_frac=outlier_frac).fit(feat_tr.values)
            reg_series = pd.Series(mdl.labels_, index=feat_tr.index)
            ret_labels = reg_series.reindex(rets_tr.index).fillna(0).astype(int).values
            p_t = mdl.predict_proba(feat_tr.values[-1:])[0]
            E = transition_matrix(mdl.labels_, n_reg)

        p_next = forecast_regime_dist(p_t, E)
        istar = int(np.argmax(p_next))
        cov = rets_tr.cov().values

        scores = {}
        if "naive" in models:
            scores["naive"] = naive_scores(rets_tr, ret_labels, istar)
        if "ridge" in models:
            kf = min(ridge_n_features, feat_tr.shape[1])
            Xr = feat_tr.reindex(rets_tr.index).iloc[:, :kf]
            scores["ridge"] = ridge_scores(Xr, rets_tr, ret_labels, p_next,
                                           feat_tr.values[-1, :kf], n_reg, alpha=ridge_alpha)
        if "mvo" in models:
            scores["mvo"] = mvo_scores(rets_tr)
        if "bl" in models:
            scores["bl"] = bl_scores(rets_tr, bl_view_means(rets_tr, ret_labels, istar))

        for mname, sc in scores.items():
            for style in MODEL_STYLES[mname]:
                for l in ls:
                    w = apply_sizing(sc, style, l, next_regime=istar)
                    w = vol_scale_weights(w, cov, vol_target)
                    records[f"{mname}_{style}_{l}"].append(float(w @ realized))
        if include_benchmarks:
            w_ew = np.ones(d) / d
            records["ew"].append(float(vol_scale_weights(w_ew, cov, vol_target) @ realized))
            w_spy = np.zeros(d)
            w_spy[spy_col] = 1.0
            records["spy"].append(float(vol_scale_weights(w_spy, cov, vol_target) @ realized))
        rec_idx.append(m)

    return pd.DataFrame(records, index=pd.DatetimeIndex(rec_idx))

