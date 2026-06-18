"""
factors/secondary_macro.py
==========================
Constructs the Secondary Macro factors following the Two Sigma Factor Lens:

    1. Emerging_Markets   – equal-risk-weighted EM equity vs DM + EM credit vs IG,
                            residualised against all four Core Macro factors.
    2. Foreign_Currency   – GDP-weighted G10 currency basket return vs USD,
                            residualised against Core Macro.
    3. Local_Inflation    – TIPS (BCIT5T) residualised vs Core Macro.
    4. Short_Volatility   – CBOE PutWrite (PUT) residualised vs Core Macro.
    5. Trend_Following    – Time-series momentum across asset classes,
                            residualised vs Core Macro.
    6. Local_Equity       – Russell 3000 (RU30INTR) residualised vs global Equity.

Foreign Currency construction
------------------------------
1. Load G10 FX spot prices (Currency_Prices.xlsx) and annual GDP by country
   (Country_GDP_in_USD.csv).
2. Convert all pairs to log returns on a USD-per-foreign basis:
       Direct  (EUR, AUD, NZD, GBP): ln(P_t / P_{t-1})
       Inverted (JPY, CHF, CAD, NOK, SEK): -ln(P_t / P_{t-1})
3. For each calendar year, compute each currency's weight as its proxy-GDP
   share of total G10 ex-USD GDP. Forward-fill to daily.
4. Compute the GDP-weighted average return across all 9 currencies.
5. Residualise the weighted basket against Core Macro factors.

All secondary factors are residualised against the provided core_macro DataFrame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from factors.core_macro import _ewm_residualise, _equal_risk_weight_combine
from data_loader import (
    load_fx_prices,
    load_gdp,
    FX_INVERT,
    FX_GDP_MAP,
)


# ── Foreign Currency ──────────────────────────────────────────────────────────

def _build_gdp_weights(
    gdp: pd.DataFrame,
    calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    For each year in the GDP table, compute each currency's share of total
    G10 ex-USD GDP.  Forward-fill to a daily calendar.

    Parameters
    ----------
    gdp      : pd.DataFrame  indexed by Year (int), columns = country names
    calendar : pd.DatetimeIndex  target business-day calendar

    Returns
    -------
    pd.DataFrame  shape (len(calendar), 9)  — daily GDP weights, one col per
    FX ticker, rows sum to 1.0.
    """
    # Build annual per-currency GDP (sum countries in each bloc)
    annual_cur_gdp: dict[str, pd.Series] = {}
    for ticker, countries in FX_GDP_MAP.items():
        available = [c for c in countries if c in gdp.columns]
        if not available:
            raise KeyError(
                f"No GDP countries found for {ticker}: need one of {countries}"
            )
        annual_cur_gdp[ticker] = gdp[available].sum(axis=1)

    annual_df = pd.DataFrame(annual_cur_gdp)  # index = Year (int)

    # Normalise rows to sum to 1
    row_sums = annual_df.sum(axis=1)
    weights_annual = annual_df.div(row_sums, axis=0)

    # Map annual weights to daily: assign year Y's weights to all days in year Y.
    # Use the GDP as known at the start of the year (no look-ahead: year Y GDP
    # is published mid-Y+1, so strictly we should lag by 1 year.  We apply a
    # 1-year lag below to keep the construction look-ahead-free.)
    weights_annual.index = pd.to_datetime(
        weights_annual.index.astype(str) + "-01-01"
    )
    # Lag by 1 year: use year Y-1 GDP for year Y weights
    weights_lagged = weights_annual.copy()
    weights_lagged.index = weights_lagged.index + pd.DateOffset(years=1)

    # Reindex to daily calendar (ffill from start of each year)
    weights_daily = weights_lagged.reindex(
        calendar, method="ffill"
    ).ffill().bfill()

    return weights_daily


def build_foreign_currency(
    returns: pd.DataFrame,
    core_macro: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    halflife_days: int = 60,
    min_periods: int = 126,
) -> pd.Series:
    """
    GDP-weighted G10 currency basket return vs USD, residualised against
    Core Macro.

    Parameters
    ----------
    returns    : pd.DataFrame  — Bloomberg log-return panel (not used directly
                 for FX; FX prices loaded fresh from disk via load_fx_prices)
    core_macro : pd.DataFrame  — Core Macro factors for residualisation
    calendar   : pd.DatetimeIndex  — business-day calendar for alignment
    halflife_days, min_periods : int

    Returns
    -------
    pd.Series  named "Foreign_Currency"
    """
    # ── 1. Load raw FX spot prices and GDP ───────────────────────────────────
    fx_prices = load_fx_prices()
    gdp       = load_gdp()

    # ── 2. Align FX prices to calendar (ffill up to 5 business days) ─────────
    fx_aligned = fx_prices.reindex(calendar).ffill(limit=5)

    # ── 3. Compute log returns, standardise to USD-per-foreign basis ──────────
    # Direct pairs (USD-per-foreign already): ln(P_t / P_{t-1})
    # Inverted pairs (foreign-per-USD):      -ln(P_t / P_{t-1})
    log_rets: dict[str, pd.Series] = {}
    for ticker in FX_GDP_MAP:
        if ticker not in fx_aligned.columns:
            continue
        prices = fx_aligned[ticker].dropna()
        lr = np.log(prices / prices.shift(1))
        if ticker in FX_INVERT:
            lr = -lr          # flip sign: rise in USD/foreign = USD weakening
        log_rets[ticker] = lr.rename(ticker)

    if not log_rets:
        raise RuntimeError("No FX return series could be computed.")

    fx_returns = pd.DataFrame(log_rets)

    # ── 4. GDP weights (lagged 1yr to avoid look-ahead) ───────────────────────
    weights = _build_gdp_weights(gdp, calendar)

    # ── 5. GDP-weighted basket return ─────────────────────────────────────────
    common_tickers = [t for t in FX_GDP_MAP if t in fx_returns.columns and t in weights.columns]
    w   = weights[common_tickers].reindex(fx_returns.index).ffill()
    ret = fx_returns[common_tickers]

    # Re-normalise weights to sum to 1 over available tickers (handles NaN periods)
    w_sum = w.sum(axis=1).replace(0, np.nan)
    w_norm = w.div(w_sum, axis=0)

    basket = (ret * w_norm).sum(axis=1, min_count=1).rename("fx_basket_raw")

    # ── 6. Residualise against Core Macro ────────────────────────────────────
    X = core_macro.dropna(how="all")
    resid = _ewm_residualise(
        basket, X, halflife_days=halflife_days, min_periods=min_periods
    )
    return resid.rename("Foreign_Currency")


# ── Trend Following helper ────────────────────────────────────────────────────

def _time_series_momentum(
    returns: pd.DataFrame,
    lookback_days: int = 252,
    rebalance_freq: str = "ME",
) -> pd.Series:
    """
    Simple time-series momentum (trend following) factor:
        - For each asset, sign of its past *lookback_days* cumulative return.
        - Equal weight across all assets.
        - Rebalance at *rebalance_freq* (default: month-end).

    Returns a daily return Series.
    """
    cumret = returns.rolling(lookback_days, min_periods=lookback_days // 2).sum()
    signal = np.sign(cumret)
    # Hold signal constant until next rebalance date
    signal = signal.resample(rebalance_freq).last().reindex(returns.index, method="ffill")
    signal = signal.fillna(0)
    n_assets = (signal != 0).sum(axis=1).replace(0, np.nan)
    equal_signal = signal.div(n_assets, axis=0)
    return (equal_signal * returns).sum(axis=1, min_count=1)


# ── Public factor constructors ────────────────────────────────────────────────

def build_emerging_markets(
    returns: pd.DataFrame,
    core_macro: pd.DataFrame,
    halflife_days: int = 60,
    min_periods: int = 126,
) -> pd.Series:
    """
    Emerging Markets factor.

    Construction
    ------------
    1. EM equity spread : MXEF (or M1EF) returns minus MXWD returns.
    2. EM credit spread : EMUSTRUU returns minus LUACTRUU returns.
    3. Equal-risk-weight the two spreads.
    4. Residualise against Core Macro factors.
    """
    if "MXEF" in returns.columns and "MXWD" in returns.columns:
        em_eq_spread = returns["MXEF"] - returns["MXWD"]
    elif "M1EF" in returns.columns and "MXWD" in returns.columns:
        em_eq_spread = returns["M1EF"] - returns["MXWD"]
    else:
        em_eq_spread = None

    if "EMUSTRUU" in returns.columns and "LUACTRUU" in returns.columns:
        em_cr_spread = returns["EMUSTRUU"] - returns["LUACTRUU"]
    elif "JPEIGLBL" in returns.columns and "LUACTRUU" in returns.columns:
        em_cr_spread = returns["JPEIGLBL"] - returns["LUACTRUU"]
    else:
        em_cr_spread = None

    if em_eq_spread is None and em_cr_spread is None:
        raise KeyError("Insufficient data to build Emerging Markets factor.")

    components = [s for s in [em_eq_spread, em_cr_spread] if s is not None]
    combined = (
        components[0].rename("em_raw")
        if len(components) == 1
        else _equal_risk_weight_combine(components).rename("em_raw")
    )

    resid = _ewm_residualise(
        combined, core_macro.dropna(how="all"),
        halflife_days=halflife_days, min_periods=min_periods,
    )
    return resid.rename("Emerging_Markets")


def build_local_inflation(
    returns: pd.DataFrame,
    core_macro: pd.DataFrame,
    halflife_days: int = 60,
    min_periods: int = 126,
) -> pd.Series:
    """
    Local Inflation factor: TIPS returns (BCIT5T) residualised against Core Macro.
    """
    if "BCIT5T" in returns.columns:
        raw = returns["BCIT5T"].rename("tips_raw")
    elif "LBUTTRUU" in returns.columns:
        raw = returns["LBUTTRUU"].rename("tips_raw")
    else:
        raise KeyError("No TIPS series (BCIT5T or LBUTTRUU) found in returns panel.")

    resid = _ewm_residualise(
        raw, core_macro.dropna(how="all"),
        halflife_days=halflife_days, min_periods=min_periods,
    )
    return resid.rename("Local_Inflation")


def build_short_volatility(
    returns: pd.DataFrame,
    core_macro: pd.DataFrame,
    halflife_days: int = 60,
    min_periods: int = 126,
) -> pd.Series:
    """
    Short Volatility factor: CBOE PutWrite Index (PUT) residualised against
    Core Macro.
    """
    if "PUT" not in returns.columns:
        raise KeyError("PUT (CBOE PutWrite) not found in returns panel.")

    resid = _ewm_residualise(
        returns["PUT"].rename("put_raw"),
        core_macro.dropna(how="all"),
        halflife_days=halflife_days, min_periods=min_periods,
    )
    return resid.rename("Short_Volatility")


def build_trend_following(
    returns: pd.DataFrame,
    core_macro: pd.DataFrame,
    lookback_days: int = 252,
    halflife_days: int = 60,
    min_periods: int = 126,
) -> pd.Series:
    """
    Trend Following factor: 12-month time-series momentum across core asset
    class series, residualised against Core Macro.
    """
    trend_assets = [
        c for c in ["MXCXDMHR", "MXWD", "LGY7TRUH", "LBUSTRUU", "LUACTRUU",
                    "BCOMTR", "SPGSCITR"]
        if c in returns.columns
    ]
    if not trend_assets:
        raise KeyError("No asset returns available to build Trend Following factor.")

    raw_trend = _time_series_momentum(
        returns[trend_assets], lookback_days=lookback_days
    ).rename("trend_raw")

    resid = _ewm_residualise(
        raw_trend, core_macro.dropna(how="all"),
        halflife_days=halflife_days, min_periods=min_periods,
    )
    return resid.rename("Trend_Following")


def build_local_equity(
    returns: pd.DataFrame,
    core_macro: pd.DataFrame,
    halflife_days: int = 60,
    min_periods: int = 126,
) -> pd.Series:
    """
    Local Equity factor: Russell 3000 (RU30INTR) residualised against Core Macro.
    Captures the US-vs-world spread in equity returns.
    """
    if "RU30INTR" not in returns.columns:
        raise KeyError("RU30INTR (Russell 3000) not found in returns panel.")

    resid = _ewm_residualise(
        returns["RU30INTR"].rename("local_eq_raw"),
        core_macro.dropna(how="all"),
        halflife_days=halflife_days, min_periods=min_periods,
    )
    return resid.rename("Local_Equity")


def build_secondary_macro(
    returns: pd.DataFrame,
    core_macro: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    halflife_days: int = 60,
    min_periods: int = 126,
    include_fx: bool = True,
    include_trend: bool = True,
    include_local_equity: bool = True,
) -> pd.DataFrame:
    """
    Build all Secondary Macro factors and return as a DataFrame.

    Parameters
    ----------
    returns      : pd.DataFrame  — daily log-return panel
    core_macro   : pd.DataFrame  — Core Macro factor DataFrame
    calendar     : pd.DatetimeIndex  — business-day calendar (needed for FX weights)
    halflife_days, min_periods : int
    include_fx   : bool  — build Foreign Currency factor (requires FX + GDP files)
    include_trend, include_local_equity : bool

    Returns
    -------
    pd.DataFrame with one column per available secondary factor.
    """
    factors: dict[str, pd.Series] = {}

    # Emerging Markets
    try:
        factors["Emerging_Markets"] = build_emerging_markets(
            returns, core_macro, halflife_days, min_periods
        )
    except KeyError as e:
        print(f"[secondary_macro] Skipping Emerging_Markets: {e}")

    # Foreign Currency
    if include_fx:
        try:
            factors["Foreign_Currency"] = build_foreign_currency(
                returns, core_macro, calendar, halflife_days, min_periods
            )
            print("[secondary_macro] Foreign_Currency: built successfully.")
        except Exception as e:
            print(f"[secondary_macro] Skipping Foreign_Currency: {e}")

    # Local Inflation
    try:
        factors["Local_Inflation"] = build_local_inflation(
            returns, core_macro, halflife_days, min_periods
        )
    except KeyError as e:
        print(f"[secondary_macro] Skipping Local_Inflation: {e}")

    # Short Volatility
    try:
        factors["Short_Volatility"] = build_short_volatility(
            returns, core_macro, halflife_days, min_periods
        )
    except KeyError as e:
        print(f"[secondary_macro] Skipping Short_Volatility: {e}")

    # Trend Following
    if include_trend:
        try:
            factors["Trend_Following"] = build_trend_following(
                returns, core_macro,
                halflife_days=halflife_days, min_periods=min_periods,
            )
        except KeyError as e:
            print(f"[secondary_macro] Skipping Trend_Following: {e}")

    # Local Equity
    if include_local_equity:
        try:
            factors["Local_Equity"] = build_local_equity(
                returns, core_macro, halflife_days, min_periods
            )
        except KeyError as e:
            print(f"[secondary_macro] Skipping Local_Equity: {e}")

    result = pd.DataFrame(factors)
    result.index.name = "Date"
    return result
