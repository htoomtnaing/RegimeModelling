"""
regime_taa.py — Core library for replicating

    "Tactical Asset Allocation with Macroeconomic Regime Detection"
    Oliveira, Sandfelder, Fujita, Dong, Cucuringu (arXiv:2503.11499v2).

Covers Stage 1 (regime classification) and the probabilistic membership layer
(Eq 1-4).  Equation numbers in docstrings refer to the paper.
"""
from __future__ import annotations

import io
import urllib.request
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
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


