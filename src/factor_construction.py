"""
factor_construction.py
======================
Orchestrates the full factor construction pipeline:

    data  →  Core Macro  →  Secondary Macro  →  Style  →  Factor Matrix

The output is a single DataFrame (the "factor matrix") where each column
is a daily return series representing one residualised factor.  This matrix
is the direct input to the GMM regime model.

Usage
-----
    from src.factor_construction import build_factor_matrix

    factor_matrix = build_factor_matrix(start="2002-01-02")
    # Returns a DataFrame with columns like:
    # [Equity, Interest_Rates, Credit, Commodities,
    #  Emerging_Markets, Local_Inflation, Short_Volatility,
    #  Trend_Following, Local_Equity,
    #  FF_SMB, FF_HML, FF_RMW, FF_CMA]
"""

from __future__ import annotations

import warnings
from typing import Optional

import pandas as pd

from src.data_cleaner import build_clean_dataset
from src.factors.core_macro import build_core_macro
from src.factors.secondary_macro import build_secondary_macro
from src.factors.style import build_all_style_factors


# ── Default factor sets for different model variants ─────────────────────────

# Minimum set for the GMM — mirrors the Two Sigma 4-cluster paper most closely.
# Start date constrained by MXCXDMHR (daily from 2002-01-02).
CORE_FACTORS = ["Equity", "Interest_Rates", "Credit", "Commodities"]

SECONDARY_FACTORS = [
    "Emerging_Markets", "Local_Inflation", "Short_Volatility",
    "Trend_Following", "Local_Equity",
]

STYLE_FACTORS = ["FF_SMB", "FF_HML", "FF_RMW", "FF_CMA"]

# Recommended start dates for different model scopes:
#   "full"    – all 9 Two Sigma-style factors (2002-01-02 limited by MXCXDMHR)
#   "reduced" – drop MXCXDMHR, use MXWD instead (1999-01-04 limited by LEGATRUU)
#   "extended"– use monthly pre-daily data where available (1987-06-30)
RECOMMENDED_STARTS = {
    "full":     "2002-01-02",
    "reduced":  "1999-01-04",
    "extended": "1987-07-01",
}


# ── Main pipeline ─────────────────────────────────────────────────────────────

def build_factor_matrix(
    start: str = "2002-01-02",
    end: Optional[str] = None,
    include_style: bool = True,
    halflife_days: int = 60,
    min_periods: int = 126,
    drop_na_threshold: float = 0.5,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build the full factor matrix used as input to the GMM regime model.

    Parameters
    ----------
    start : str
        Start date for the factor matrix.  Use RECOMMENDED_STARTS for
        guidance on which start date matches available daily data.
    end : str or None
        End date.  Defaults to today.
    include_style : bool
        If True, appends FF5-based style factors (SMB, HML, RMW, CMA)
        after residualising against Core Macro.
    halflife_days : int
        EWM half-life (trading days) for rolling residualisation.
    min_periods : int
        Minimum observations before the first residual is computed.
        Rows before this threshold will be NaN for residualised factors.
    drop_na_threshold : float
        Drop rows where more than this fraction of columns are NaN.
        E.g. 0.5 drops rows with >50% missing factors.
    verbose : bool
        Print progress information.

    Returns
    -------
    pd.DataFrame  shape (T, N_factors)
        Clean, aligned factor return matrix indexed by date.
    """
    if verbose:
        print(f"[factor_construction] Building factor matrix: {start} → {end or 'today'}")

    # ── 1. Load and clean raw data ──────────────────────────────────────────
    if verbose:
        print("  Loading raw data...")
    dataset  = build_clean_dataset(start=start, end=end)
    returns  = dataset["returns"]
    ff5      = dataset["ff5"]
    calendar = dataset["calendar"]

    if verbose:
        print(f"  Returns panel: {returns.shape[0]} days × {returns.shape[1]} series")
        print(f"  FF5 panel:     {ff5.shape[0]} days × {ff5.shape[1]} factors")

    # ── 2. Core Macro factors ──────────────────────────────────────────────
    if verbose:
        print("  Building Core Macro factors (Equity, Rates, Credit, Commodities)...")
    core_macro = build_core_macro(
        returns,
        halflife_days=halflife_days,
        min_periods=min_periods,
    )
    if verbose:
        _core_valid = core_macro.dropna(how="any")
        available_from = _core_valid.index[0].date() if not _core_valid.empty else "N/A (all NaN — check residualiser)"
        print(f"  Core Macro: all 4 factors valid from {available_from}")

    # ── 3. Secondary Macro factors ─────────────────────────────────────────
    if verbose:
        print("  Building Secondary Macro factors...")
    with warnings.catch_warnings(record=True) as w_list:
        warnings.simplefilter("always")
        secondary_macro = build_secondary_macro(
            returns,
            core_macro,
            calendar=calendar,
            halflife_days=halflife_days,
            min_periods=min_periods,
        )
    for w in w_list:
        if verbose:
            print(f"  [warn] {w.message}")

    if verbose:
        print(f"  Secondary Macro columns: {list(secondary_macro.columns)}")

    # ── 4. Style factors (optional) ────────────────────────────────────────
    if include_style:
        if verbose:
            print("  Building Style factors (FF5 residualised vs Core Macro)...")
        style = build_all_style_factors(
            ff5, core_macro,
            halflife_days=halflife_days,
            min_periods=min_periods,
        )
        if verbose:
            print(f"  Style columns: {list(style.columns)}")
    else:
        style = pd.DataFrame(index=returns.index)

    # ── 5. Combine into factor matrix ─────────────────────────────────────
    parts = [core_macro, secondary_macro]
    if include_style and not style.empty:
        parts.append(style)

    factor_matrix = pd.concat(parts, axis=1)
    factor_matrix = factor_matrix.sort_index()

    # ── 6. Quality filtering ───────────────────────────────────────────────
    # Drop rows where the fraction of NaN columns exceeds drop_na_threshold.
    # During the min_periods burn-in (~126 days) residualised factors are NaN,
    # so we use a generous threshold — the default 0.5 means a row must have
    # at least half its factors valid to be kept.
    n_before = len(factor_matrix)
    max_na_cols = int(drop_na_threshold * factor_matrix.shape[1])
    factor_matrix = factor_matrix[factor_matrix.isna().sum(axis=1) <= max_na_cols]
    n_after = len(factor_matrix)

    if verbose:
        print(f"  After NA filtering (threshold={drop_na_threshold}): "
              f"{n_after} rows kept, {n_before - n_after} dropped")
        if n_after == 0:
            print("  WARNING: 0 rows survived NA filtering!")
            print("  NaN counts per factor before filtering:")
            full_fm = pd.concat([core_macro, secondary_macro] +
                                ([style] if include_style and not style.empty else []), axis=1)
            print(full_fm.isna().sum().to_string())
        else:
            print(f"  Factor matrix shape: {factor_matrix.shape}")
            print(f"  Date range: {factor_matrix.index[0].date()} -> {factor_matrix.index[-1].date()}")
            print(f"  Columns: {list(factor_matrix.columns)}")

    return factor_matrix


def get_factor_matrix_for_gmm(
    factor_matrix: pd.DataFrame,
    factors: Optional[list[str]] = None,
    dropna: bool = True,
) -> pd.DataFrame:
    """
    Extract and clean a sub-matrix from the full factor_matrix for GMM fitting.

    Parameters
    ----------
    factor_matrix : pd.DataFrame
        Output of build_factor_matrix().
    factors : list[str] or None
        Column subset to use.  If None, uses all columns.
    dropna : bool
        If True, drops any remaining rows with NaN values (required for GMM).

    Returns
    -------
    pd.DataFrame  — clean matrix with no NaN values.
    """
    if factors is not None:
        missing = [f for f in factors if f not in factor_matrix.columns]
        if missing:
            raise KeyError(f"Factors not found in matrix: {missing}")
        sub = factor_matrix[factors].copy()
    else:
        sub = factor_matrix.copy()

    if dropna:
        sub = sub.dropna()

    return sub
