"""
factors/style.py
================
Constructs equity style factors for the regime model.

Sources
-------
- Fama-French 5 factors (daily): Mkt-RF, SMB, HML, RMW, CMA
  These are already long-short portfolios and are in *decimal* form
  (divided by 100 in the loader).

- The style factors are residualised against the Core Macro factors to
  isolate the pure stock-selection / style premia from the macro beta.

Factors produced
----------------
    FF_Market    – Mkt-RF (market excess return, already vs RF)
    FF_SMB       – Small minus Big (size premium)
    FF_HML       – High minus Low (value premium)
    FF_RMW       – Robust minus Weak (profitability premium)
    FF_CMA       – Conservative minus Aggressive (investment premium)

All are residualised against Core Macro by default (matching Two Sigma's
approach of making each factor orthogonal to the macro factors).
"""

from __future__ import annotations

import pandas as pd

from src.factors.core_macro import _ewm_residualise


# ── Public factor constructors ────────────────────────────────────────────────

def build_ff5_factors(
    ff5: pd.DataFrame,
    core_macro: pd.DataFrame,
    residualise: bool = True,
    halflife_days: int = 60,
    min_periods: int = 126,
) -> pd.DataFrame:
    """
    Build FF5-derived style factors, optionally residualised against Core Macro.

    Parameters
    ----------
    ff5 : pd.DataFrame
        Daily FF5 returns in decimal from data_cleaner.align_ff5().
        Expected columns: Mkt_RF, SMB, HML, RMW, CMA, RF.
    core_macro : pd.DataFrame
        Core Macro factor DataFrame (Equity, Interest_Rates, Credit, Commodities).
    residualise : bool
        If True (default), residualise each factor against Core Macro.
        Set False to get raw FF5 factors (useful for comparison).
    halflife_days, min_periods : int
        Passed to _ewm_residualise.

    Returns
    -------
    pd.DataFrame with columns prefixed FF_ (e.g. FF_SMB, FF_HML, ...).
    """
    style_cols = {
        "Mkt_RF": "FF_Market",
        "SMB":    "FF_SMB",
        "HML":    "FF_HML",
        "RMW":    "FF_RMW",
        "CMA":    "FF_CMA",
    }

    available = {k: v for k, v in style_cols.items() if k in ff5.columns}
    if not available:
        raise KeyError("No FF5 columns found in ff5 DataFrame.")

    X = core_macro.dropna(how="all")
    results: dict[str, pd.Series] = {}

    for raw_col, factor_name in available.items():
        raw = ff5[raw_col].rename(raw_col)
        if residualise:
            factor = _ewm_residualise(raw, X, halflife_days=halflife_days, min_periods=min_periods)
        else:
            factor = raw.copy()
        results[factor_name] = factor.rename(factor_name)

    out = pd.DataFrame(results)
    out.index.name = "Date"
    return out


def build_all_style_factors(
    ff5: pd.DataFrame,
    core_macro: pd.DataFrame,
    halflife_days: int = 60,
    min_periods: int = 126,
) -> pd.DataFrame:
    """
    Convenience wrapper — returns all available style factors residualised
    against Core Macro.
    """
    return build_ff5_factors(
        ff5, core_macro,
        residualise=True,
        halflife_days=halflife_days,
        min_periods=min_periods,
    )
