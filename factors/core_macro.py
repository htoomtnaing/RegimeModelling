"""
factors/core_macro.py
=====================
Constructs the four Core Macro factors following the Two Sigma Factor Lens:

    1. Equity         – MSCI ACWI hedged USD log returns
    2. Interest Rates – Global Govt 7-10yr hedged USD log returns
    3. Credit         – Residual of equal-risk-weighted US IG + US HY + EU IG + EU HY
                        vs Equity + Rates
    4. Commodities    – Residual of Bloomberg Commodity TR vs Equity + Rates

Residualisation uses a rolling exponentially-weighted OLS (60-day half-life).

Performance note
----------------
The original per-row Python loop was O(T²) and timed out on 6,000+ row datasets.
This version uses a vectorised recursive EWM update of the sufficient statistics
(XᵀWX and XᵀWy) which is O(T·N²) — fast enough for daily data over decades.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Vectorised EWM-OLS residualiser ──────────────────────────────────────────

def _ewm_residualise(
    y: pd.Series,
    X: pd.DataFrame,
    halflife_days: int = 60,
    min_periods: int = 126,
) -> pd.Series:
    """
    Vectorised rolling exponentially-weighted OLS residualiser.

    At each time t, fit weighted OLS of y ~ [1, X] using all observations
    up to t-1, with exponential weights (decay = ln2 / halflife_days).
    Returns the one-step-ahead residual: y_t - X_t @ beta_{t-1}.

    This is look-ahead-free: beta is estimated using data strictly before t.

    Implementation
    --------------
    Uses a recursive update of the EWM sufficient statistics:
        A_t = λ·A_{t-1} + w_t · x_t x_tᵀ       (XᵀWX)
        b_t = λ·b_{t-1} + w_t · x_t y_t         (XᵀWy)
    where λ = exp(-ln2 / halflife_days) is the per-step decay factor.
    At each step, beta_t = A_t⁻¹ b_t, and the residual is computed for t+1.

    Complexity: O(T · K²) where K = number of regressors — fast even for T=10,000.
    """
    # Align on common index — drop NaN from y first so the residualiser
    # never receives NaN dependent-variable values (which corrupt A and b).
    y_clean = y.dropna()
    common  = y_clean.index.intersection(X.dropna().index)
    y_s = y_clean.loc[common].values.astype(float)
    X_s = X.loc[common].values.astype(float)
    T, K = X_s.shape

    # Add intercept column
    Xc = np.column_stack([np.ones(T), X_s])  # shape (T, K+1)
    Kc = Xc.shape[1]

    decay = np.log(2) / halflife_days          # per-step exponential decay rate
    lam   = np.exp(-decay)                     # multiplier applied to old stats each step

    # Initialise sufficient statistics (small ridge for numerical stability)
    ridge = 1e-8
    A = np.eye(Kc) * ridge   # XᵀWX  (Kc × Kc)
    b = np.zeros(Kc)          # XᵀWy  (Kc,)

    resid = np.full(T, np.nan)

    for t in range(T):
        # --- Predict at t using beta estimated from [0..t-1] ---
        if t >= min_periods:
            try:
                beta = np.linalg.solve(A, b)
                resid[t] = y_s[t] - Xc[t] @ beta
            except np.linalg.LinAlgError:
                pass   # leave as NaN if singular

        # --- Update sufficient statistics with observation t ---
        A = lam * A + np.outer(Xc[t], Xc[t])
        b = lam * b + Xc[t] * y_s[t]

    return pd.Series(resid, index=common, name=y.name)


def _equal_risk_weight_combine(
    series_list: list[pd.Series],
    window: int = 252,
) -> pd.Series:
    """
    Combine multiple return series using equal risk weights (inverse volatility).
    Volatility estimated on a rolling *window*-day basis.
    """
    df = pd.concat(series_list, axis=1).dropna(how="all")
    vols    = df.rolling(window, min_periods=window // 2).std()
    inv_vol = 1.0 / vols
    weights = inv_vol.div(inv_vol.sum(axis=1), axis=0)
    return (df * weights).sum(axis=1, min_count=1)


# ── Public factor constructors ────────────────────────────────────────────────

def build_equity(returns: pd.DataFrame) -> pd.Series:
    """
    Equity factor: MSCI ACWI hedged USD (MXCXDMHR) log returns.
    Falls back to MXWD (unhedged) if MXCXDMHR is unavailable.
    """
    if "MXCXDMHR" in returns.columns:
        return returns["MXCXDMHR"].rename("Equity")
    elif "MXWD" in returns.columns:
        return returns["MXWD"].rename("Equity")
    raise KeyError("Neither MXCXDMHR nor MXWD found in returns panel.")


def build_interest_rates(returns: pd.DataFrame) -> pd.Series:
    """
    Interest Rates factor: Global Govt 7-10yr hedged USD (LGY7TRUH) log returns.
    Falls back to LBUSTRUU (US Agg) if LGY7TRUH is unavailable.
    """
    if "LGY7TRUH" in returns.columns:
        return returns["LGY7TRUH"].rename("Interest_Rates")
    elif "LBUSTRUU" in returns.columns:
        return returns["LBUSTRUU"].rename("Interest_Rates")
    raise KeyError("Neither LGY7TRUH nor LBUSTRUU found in returns panel.")


def build_credit(
    returns: pd.DataFrame,
    equity: pd.Series,
    interest_rates: pd.Series,
    halflife_days: int = 60,
    min_periods: int = 126,
) -> pd.Series:
    """
    Credit factor: equal-risk-weighted combination of up to four credit
    sub-indices, residualised against Equity + Interest Rates.

    Sub-indices (Two Sigma Factor Lens construction):
        LUACTRUU  – US Corporate IG          (required)
        LF98TRUU  – US Corporate HY          (required)
        LP05TRUH  – Pan Euro Agg Corp USD    (optional, EU IG leg)
        LP01TRUH  – Pan Euro HY USD          (optional, EU HY leg)

    EU legs phase in from mid-2000 when their daily data begins.
    Falls back gracefully to US-only if EU files are absent.
    """
    required = ["LUACTRUU", "LF98TRUU"]
    optional = ["LP05TRUH", "LP01TRUH"]

    missing = [t for t in required if t not in returns.columns]
    if missing:
        raise KeyError(f"Required credit series missing: {missing}")

    available = required + [t for t in optional if t in returns.columns]
    n_eu = sum(1 for t in optional if t in returns.columns)
    if n_eu > 0:
        print(f"[core_macro] Credit: {len(available)} sub-indices "
              f"(US IG, US HY, EU IG, EU HY) — EU legs active from mid-2000.")
    else:
        print("[core_macro] Credit: EU legs not found — using US IG + US HY only.")

    combined = _equal_risk_weight_combine(
        [returns[t] for t in available]
    ).rename("credit_raw")

    X = pd.concat([equity, interest_rates], axis=1).dropna()
    resid = _ewm_residualise(combined, X, halflife_days=halflife_days, min_periods=min_periods)
    return resid.rename("Credit")


def build_commodities(
    returns: pd.DataFrame,
    equity: pd.Series,
    interest_rates: pd.Series,
    halflife_days: int = 60,
    min_periods: int = 126,
) -> pd.Series:
    """
    Commodities factor: Bloomberg Commodity TR (BCOMTR) residualised against
    Equity + Interest Rates. Falls back to SPGSCITR if BCOMTR unavailable.
    """
    if "BCOMTR" in returns.columns:
        raw = returns["BCOMTR"].rename("commod_raw")
    elif "SPGSCITR" in returns.columns:
        raw = returns["SPGSCITR"].rename("commod_raw")
    else:
        raise KeyError("Neither BCOMTR nor SPGSCITR found in returns panel.")

    X = pd.concat([equity, interest_rates], axis=1).dropna()
    resid = _ewm_residualise(raw, X, halflife_days=halflife_days, min_periods=min_periods)
    return resid.rename("Commodities")


def build_core_macro(
    returns: pd.DataFrame,
    halflife_days: int = 60,
    min_periods: int = 126,
) -> pd.DataFrame:
    """
    Build all four Core Macro factors and return as a DataFrame.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily log-return panel from data_cleaner.build_return_panel().
    halflife_days : int
        EWM half-life for the rolling residualisation.
    min_periods : int
        Minimum observations before the first residual is computed.

    Returns
    -------
    pd.DataFrame with columns [Equity, Interest_Rates, Credit, Commodities].
    """
    equity      = build_equity(returns)
    rates       = build_interest_rates(returns)
    credit      = build_credit(returns, equity, rates, halflife_days, min_periods)
    commodities = build_commodities(returns, equity, rates, halflife_days, min_periods)

    core = pd.concat([equity, rates, credit, commodities], axis=1)
    core.index.name = "Date"
    return core
