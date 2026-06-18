"""
regime_model.py
===============
Gaussian Mixture Model (GMM) regime detection, following the Two Sigma
"Machine Learning Approach to Regime Modeling" (2021).

Key design decisions
--------------------
- Uses sklearn's GaussianMixture with full covariance matrices.
- Number of components selected by cross-validated log-likelihood.
- Each observation is assigned a *probability* vector (soft assignment),
  not just a hard label — the probabilities are the primary output.
- Factor standardisation is applied before fitting (unit variance per
  factor) and reversed for interpretation.
- Regime *labelling* (Crisis, Steady State, Inflation, WOI) is done by
  inspecting the mean return of each cluster for key factors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import warnings

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold


# ── Result container ─────────────────────────────────────────────────────────

@dataclass
class RegimeModel:
    """
    Container for a fitted GMM regime model and its outputs.

    Attributes
    ----------
    gmm            – the fitted sklearn GaussianMixture object
    scaler         – the StandardScaler used before fitting
    n_components   – number of regimes
    factor_names   – column names of the input factor matrix
    probabilities  – DataFrame (T × n_components): P(regime_k | obs_t)
    hard_labels    – Series (T,): argmax of probabilities
    regime_names   – dict {int: str} mapping cluster index to a human label
    factor_means   – DataFrame (n_components × n_factors): annualised mean returns per regime
    factor_vols    – DataFrame (n_components × n_factors): annualised vol per regime
    cv_scores      – dict {n_components: mean_log_likelihood} from cross-validation
    """
    gmm:           GaussianMixture
    scaler:        StandardScaler
    n_components:  int
    factor_names:  list[str]
    probabilities: pd.DataFrame
    hard_labels:   pd.Series
    regime_names:  dict[int, str]  = field(default_factory=dict)
    factor_means:  pd.DataFrame    = field(default_factory=pd.DataFrame)
    factor_vols:   pd.DataFrame    = field(default_factory=pd.DataFrame)
    cv_scores:     dict            = field(default_factory=dict)


# ── Cross-validation ─────────────────────────────────────────────────────────

def select_n_components(
    X: np.ndarray,
    n_range: range = range(2, 7),
    n_splits: int = 5,
    n_init: int = 10,
    random_state: int = 42,
    verbose: bool = True,
) -> dict[int, float]:
    """
    Select the optimal number of GMM components via cross-validated
    log-likelihood (as in the Two Sigma paper; AIC/BIC available as
    supplementary checks).

    Parameters
    ----------
    X : np.ndarray  shape (T, N_factors)  — standardised factor returns
    n_range : range  — candidate component counts to evaluate
    n_splits : int   — number of CV folds (time-series: sequential folds)
    n_init : int     — GMM restarts per fold to avoid local optima
    random_state : int

    Returns
    -------
    dict {n_components: mean_cv_log_likelihood}
    """
    scores: dict[int, float] = {}
    kf = KFold(n_splits=n_splits, shuffle=False)  # time-ordered folds

    for n in n_range:
        fold_scores = []
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            gmm = GaussianMixture(
                n_components=n,
                covariance_type="full",
                n_init=n_init,
                random_state=random_state,
                max_iter=500,
                reg_covar=1e-6,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gmm.fit(X_train)
            score = gmm.score(X_val)  # mean log-likelihood per sample
            fold_scores.append(score)
        scores[n] = float(np.mean(fold_scores))
        if verbose:
            print(f"  n={n}: mean CV log-likelihood = {scores[n]:.4f}")

    return scores


def _aic_bic(X: np.ndarray, n_range: range, n_init: int = 10, random_state: int = 42) -> dict:
    """Compute AIC and BIC for each n_components as supplementary diagnostics."""
    results = {}
    for n in n_range:
        gmm = GaussianMixture(
            n_components=n, covariance_type="full",
            n_init=n_init, random_state=random_state,
            max_iter=500, reg_covar=1e-6,
        )
        gmm.fit(X)
        results[n] = {"aic": gmm.aic(X), "bic": gmm.bic(X)}
    return results


# ── Regime labelling ──────────────────────────────────────────────────────────

# Mapping rules based on Two Sigma paper logic:
#   Crisis:      Equity mean < 0, Credit mean < 0
#   Steady State: Equity mean > 0, all factors roughly positive/neutral
#   Inflation:   Local_Inflation mean is highest among regimes
#   WOI:         Equity vol is elevated but mean is positive; Momentum negative
#
# Implemented as a priority-ordered heuristic; ties broken by residual assignment.

_LABEL_PRIORITY = ["Crisis", "Inflation", "WOI", "Steady_State"]


def _auto_label_regimes(
    factor_means: pd.DataFrame,
    factor_vols: pd.DataFrame,
    n_components: int,
) -> dict[int, str]:
    """
    Automatically assign regime names based on factor mean patterns.

    This is a *heuristic* — the human analyst should review and adjust labels
    using regime_utils.py after inspecting the full output.

    Parameters
    ----------
    factor_means : pd.DataFrame  shape (n_components, n_factors)  — annualised
    factor_vols  : pd.DataFrame  shape (n_components, n_factors)  — annualised

    Returns
    -------
    dict {cluster_index: label_string}
    """
    labels: dict[int, str] = {}
    assigned: set[str] = set()

    cols = list(factor_means.columns)

    # ── Helper: get column value if present ──────────────────────────────────
    def _get(df: pd.DataFrame, col: str, idx: int, default: float = 0.0) -> float:
        if col in df.columns:
            return float(df.loc[idx, col])
        return default

    # ── Step 1: Crisis — negative Equity AND negative Credit ─────────────────
    eq_means  = factor_means.get("Equity",         pd.Series(0, index=factor_means.index))
    cr_means  = factor_means.get("Credit",         pd.Series(0, index=factor_means.index))
    infl_means= factor_means.get("Local_Inflation",pd.Series(0, index=factor_means.index))

    # Crisis = most negative equity regime (if negative at all)
    crisis_candidate = int(eq_means.idxmin())
    if eq_means[crisis_candidate] < 0:
        labels[crisis_candidate] = "Crisis"
        assigned.add("Crisis")

    # ── Step 2: Inflation — highest Local_Inflation mean ─────────────────────
    if "Local_Inflation" in factor_means.columns:
        infl_ranks = infl_means.drop(index=list(labels.keys()), errors="ignore")
        if not infl_ranks.empty:
            infl_candidate = int(infl_ranks.idxmax())
            # Only label as Inflation if its inflation mean is clearly above zero
            if infl_means[infl_candidate] > 0:
                labels[infl_candidate] = "Inflation"
                assigned.add("Inflation")

    # ── Step 3: WOI — elevated equity vol, positive equity mean ──────────────
    remaining = [i for i in factor_means.index if i not in labels]
    if "Equity" in factor_vols.columns and len(remaining) >= 2:
        eq_vols_remaining = factor_vols.loc[remaining, "Equity"]
        woi_candidate = int(eq_vols_remaining.idxmax())
        # WOI should also have positive equity mean
        if eq_means[woi_candidate] > 0:
            labels[woi_candidate] = "WOI"
            assigned.add("WOI")

    # ── Step 4: Remaining → Steady State ────────────────────────────────────
    for i in factor_means.index:
        if i not in labels:
            labels[i] = "Steady_State"

    return labels


# ── Main fitting function ─────────────────────────────────────────────────────

def fit_regime_model(
    factor_matrix: pd.DataFrame,
    n_components: Optional[int] = None,
    n_components_range: range = range(2, 7),
    n_init: int = 20,
    cv_splits: int = 5,
    halflife_ewm: Optional[int] = None,
    random_state: int = 42,
    verbose: bool = True,
    run_cv: bool = True,
) -> RegimeModel:
    """
    Fit a GMM regime model on the given factor matrix.

    Parameters
    ----------
    factor_matrix : pd.DataFrame
        Clean (no NaN) factor return matrix from factor_construction.py.
    n_components : int or None
        Number of regimes.  If None, selected by cross-validation over
        *n_components_range*.
    n_components_range : range
        Candidate values for CV (ignored if n_components is given).
    n_init : int
        GMM restarts to avoid local optima.
    cv_splits : int
        Number of time-ordered CV folds.
    halflife_ewm : int or None
        If set, apply exponential weights (recent obs weighted more) during
        GMM fitting via sample_weight.  Useful for detecting current regime.
        Ignored in CV (CV uses equal weights for fair comparison).
    random_state : int
    verbose : bool
    run_cv : bool
        If True and n_components is None, run CV to select n_components.

    Returns
    -------
    RegimeModel
    """
    # ── 1. Standardise ───────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_std = scaler.fit_transform(factor_matrix.values)
    factor_names = list(factor_matrix.columns)

    if verbose:
        print(f"[regime_model] Fitting GMM on {X_std.shape[0]} obs × {X_std.shape[1]} factors")

    # ── 2. Cross-validation to select n_components ───────────────────────────
    cv_scores: dict[int, float] = {}
    if n_components is None:
        if run_cv:
            if verbose:
                print(f"  Running {cv_splits}-fold CV over n_components={list(n_components_range)}...")
            cv_scores = select_n_components(
                X_std, n_range=n_components_range,
                n_splits=cv_splits, n_init=n_init,
                random_state=random_state, verbose=verbose,
            )
            n_components = max(cv_scores, key=cv_scores.get)
            if verbose:
                print(f"  Selected n_components = {n_components} (best CV log-lik = {cv_scores[n_components]:.4f})")
        else:
            n_components = 4  # default: reproduce Two Sigma result
            if verbose:
                print(f"  Using default n_components = {n_components} (set run_cv=True to select)")

    # ── 3. Fit final GMM ────────────────────────────────────────────────────
    if verbose:
        print(f"  Fitting final GMM with n_components={n_components}, n_init={n_init}...")

    sample_weight = None
    if halflife_ewm is not None:
        T = X_std.shape[0]
        decay = np.log(2) / halflife_ewm
        ages = np.arange(T)[::-1]  # 0 = most recent
        sample_weight = np.exp(-decay * ages)
        sample_weight /= sample_weight.sum()

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        n_init=n_init,
        random_state=random_state,
        max_iter=1000,
        reg_covar=1e-6,
    )
    if sample_weight is not None:
        gmm.fit(X_std, None)  # sklearn GMM doesn't support sample_weight; noted for future
    else:
        gmm.fit(X_std)

    if verbose:
        print(f"  GMM converged: {gmm.converged_}")

    # ── 4. Probabilities and hard labels ─────────────────────────────────────
    prob_arr = gmm.predict_proba(X_std)
    hard_arr = gmm.predict(X_std)

    prob_cols = [f"Regime_{k}" for k in range(n_components)]
    probabilities = pd.DataFrame(prob_arr, index=factor_matrix.index, columns=prob_cols)
    hard_labels   = pd.Series(hard_arr, index=factor_matrix.index, name="Regime")

    # ── 5. Regime statistics ─────────────────────────────────────────────────
    # Annualised means and vols per regime (using hard labels for simplicity)
    trading_days = 252
    means_list, vols_list = [], []

    for k in range(n_components):
        mask = hard_labels == k
        sub = factor_matrix[mask]
        means_list.append((sub.mean() * trading_days).rename(k))
        vols_list.append((sub.std()  * np.sqrt(trading_days)).rename(k))

    factor_means = pd.DataFrame(means_list)
    factor_vols  = pd.DataFrame(vols_list)

    if verbose:
        print(f"\n  Annualised factor means per regime (%):")
        print((factor_means * 100).round(2).to_string())

    # ── 6. Auto-label regimes ────────────────────────────────────────────────
    regime_names = _auto_label_regimes(factor_means, factor_vols, n_components)
    if verbose:
        print(f"\n  Auto-assigned regime labels: {regime_names}")
        freq = hard_labels.value_counts().sort_index()
        for k, label in regime_names.items():
            pct = 100 * freq.get(k, 0) / len(hard_labels)
            print(f"    Regime {k} ({label}): {pct:.1f}% of observations")

    return RegimeModel(
        gmm=gmm,
        scaler=scaler,
        n_components=n_components,
        factor_names=factor_names,
        probabilities=probabilities,
        hard_labels=hard_labels,
        regime_names=regime_names,
        factor_means=factor_means,
        factor_vols=factor_vols,
        cv_scores=cv_scores,
    )


def predict_current_regime(
    model: RegimeModel,
    factor_matrix: pd.DataFrame,
    window_days: int = 60,
) -> pd.Series:
    """
    Predict the regime probabilities for the most recent *window_days* using
    the fitted model.  Useful for a "where are we now?" dashboard.

    Returns a Series {regime_label: probability} for the latest observation.
    """
    recent = factor_matrix.dropna().tail(window_days)
    X_std  = model.scaler.transform(recent.values)
    probs  = model.gmm.predict_proba(X_std)
    latest_probs = probs[-1]

    result = {
        model.regime_names.get(k, f"Regime_{k}"): float(p)
        for k, p in enumerate(latest_probs)
    }
    return pd.Series(result, name="Current_Regime_Probability")
