"""
data_cleaner.py
===============
Transforms raw price series (from data_loader.py) into clean, aligned,
daily log-return DataFrames ready for factor construction.

Key responsibilities
--------------------
1. Build a master business-day calendar.
2. Reindex every series to that calendar and forward-fill (up to a cap).
3. Handle the monthly→daily transitions: before the daily-from date the
   series is at monthly frequency; after, it is daily.  We keep the full
   history but use a longer ffill cap in the monthly section.
4. Compute log returns: ln(P_t / P_{t-1}).
5. Build a risk-free rate (daily decimal) by stitching Fed Funds + T-bill.
6. Remove zero-return days that are artifacts of repeated prices (holidays
   in bond indices) by forward-filling the price instead.
"""

from __future__ import annotations
from typing import Optional

import numpy as np
import pandas as pd

from data_loader import (
    DAILY_FROM,
    load_bloomberg,
    load_bloomberg_all,
    load_ff5_daily,
    load_interest_rate_daily,
    load_cpi_interest_rates,
    ALL_BLOOMBERG_TICKERS,
)

# ── Constants ────────────────────────────────────────────────────────────────

# Maximum business days to forward-fill in the DAILY section.
# Bond indices may carry prices over short holidays — 5 days is safe.
FFILL_DAILY_LIMIT = 5

# Maximum calendar days to forward-fill in the MONTHLY section.
# 35 days covers month-end to next month-end with some slack.
FFILL_MONTHLY_LIMIT = 35

# Tickers that are known to have repeated prices caused by holiday carries
# in the daily series (BCIT5T is the worst at 4.1%).
HOLIDAY_CARRY_TICKERS = {"BCIT5T", "LUACTRUU", "LF98TRUU", "LP05TRUH", "LP01TRUH", "LBUSTRUU", "PUT", "EMUSTRUU", "MXCXDMHR"}


# ── Master calendar ──────────────────────────────────────────────────────────

def build_calendar(start: str = "1987-01-02", end: Optional[str] = None) -> pd.DatetimeIndex:
    """
    Return a DatetimeIndex of business days from *start* to *end* (inclusive).
    We use US business days (Mon-Fri) as the universal anchor; non-US holidays
    are handled by forward-filling at the reindex step.
    """
    if end is None:
        end = pd.Timestamp.today().normalize()
    return pd.bdate_range(start=start, end=end, freq="B")


# ── Core cleaning helpers ────────────────────────────────────────────────────

def _strip_holiday_carries(prices: pd.Series) -> pd.Series:
    """
    Replace zero-diff prices (i.e. price_{t} == price_{t-1}) with NaN so
    that subsequent ffill treats them as missing rather than returning a
    spurious zero log-return.  Only applied to tickers in HOLIDAY_CARRY_TICKERS.
    """
    p = prices.copy().astype(float)
    dupe_mask = p.diff() == 0
    p[dupe_mask] = np.nan
    return p


def align_to_calendar(
    raw: pd.Series,
    calendar: pd.DatetimeIndex,
    ticker: str = "",
) -> pd.Series:
    """
    Reindex *raw* (a price Series with potentially mixed frequency) to
    *calendar*, forward-filling gaps sensibly:

    - Before DAILY_FROM[ticker] (if it exists): monthly data, ffill up to
      FFILL_MONTHLY_LIMIT calendar days.
    - After DAILY_FROM[ticker] (or always, if no transition): daily data,
      ffill up to FFILL_DAILY_LIMIT business days.

    Returns a Series on the *calendar* index.
    """
    raw = raw.sort_index()
    daily_from = pd.Timestamp(DAILY_FROM[ticker]) if ticker in DAILY_FROM else None

    if daily_from is None or raw.index[0] >= daily_from:
        # Purely daily series
        s = raw.reindex(calendar).ffill(limit=FFILL_DAILY_LIMIT)
        if ticker in HOLIDAY_CARRY_TICKERS:
            s = _strip_holiday_carries(s).ffill(limit=FFILL_DAILY_LIMIT)
        return s

    # Split into monthly and daily sections
    monthly_mask = calendar < daily_from
    daily_mask   = calendar >= daily_from

    cal_monthly = calendar[monthly_mask]
    cal_daily   = calendar[daily_mask]

    raw_monthly = raw[raw.index < daily_from]
    raw_daily   = raw[raw.index >= daily_from]

    s_monthly = raw_monthly.reindex(cal_monthly, method=None)
    # ffill monthly obs across the (empty) business days between month-ends
    s_monthly = s_monthly.ffill(limit=FFILL_MONTHLY_LIMIT)

    s_daily = raw_daily.reindex(cal_daily).ffill(limit=FFILL_DAILY_LIMIT)
    if ticker in HOLIDAY_CARRY_TICKERS:
        s_daily = _strip_holiday_carries(s_daily).ffill(limit=FFILL_DAILY_LIMIT)

    return pd.concat([s_monthly, s_daily]).rename(raw.name)


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """Compute log returns: ln(P_t / P_{t-1})."""
    return np.log(prices / prices.shift(1)).rename(prices.name)


# ── Risk-free rate ───────────────────────────────────────────────────────────

def build_rf_daily(calendar: pd.DatetimeIndex) -> pd.Series:
    """
    Build a daily risk-free rate Series (decimal, e.g. 0.0001 per day) on
    *calendar* by stitching the best available FRED series:

        dff  (Fed Funds Rate, 1954-)   → primary source, annualised %
        dtb3 (3m T-bill,     1954-)    → secondary
        dgs3mo (3m CMT,      1981-)    → tertiary

    The annualised rate r% is converted to a daily rate via:
        daily_rf = (1 + r/100)^(1/252) - 1

    Before 1954 we fall back to the quarterly t30ret from CPI_InterestRates.
    """
    ir = load_interest_rate_daily()

    # Prefer dff; fill gaps with dtb3 then dgs3mo
    rf_ann = (
        ir["dff"]
        .combine_first(ir["dtb3"])
        .combine_first(ir["dgs3mo"])
    )

    # Extend back with quarterly t30ret from CPI file (it is a period return,
    # not an annualised rate, so annualise differently)
    cpi = load_cpi_interest_rates()
    # t30ret is the quarterly total return on 30-day T-bills → annualise
    # approximate: annual_r ≈ (1 + quarterly_ret)^4 - 1  → daily
    if "t30ret" in cpi.columns:
        t30_annual_pct = ((1 + cpi["t30ret"]) ** 4 - 1) * 100
        t30_annual_pct = t30_annual_pct.reindex(calendar, method="ffill")
        rf_ann = rf_ann.combine_first(t30_annual_pct.rename("dff"))

    # Reindex to calendar
    rf_ann_cal = rf_ann.reindex(calendar).ffill(limit=10)

    # Convert annualised % → daily decimal
    rf_daily = (1 + rf_ann_cal / 100) ** (1 / 252) - 1
    rf_daily.name = "RF"
    return rf_daily


# ── FF5 alignment ────────────────────────────────────────────────────────────

def align_ff5(calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Load the FF5 daily factors and reindex to *calendar*.
    Returns decimal returns (already divided by 100 in the loader).
    """
    ff5 = load_ff5_daily()
    return ff5.reindex(calendar).ffill(limit=FFILL_DAILY_LIMIT)


# ── Master pipeline ──────────────────────────────────────────────────────────

def build_price_panel(
    tickers: list[str] | None = None,
    calendar: pd.DatetimeIndex | None = None,
    start: str = "1987-01-02",
    end: str | None = None,
) -> pd.DataFrame:
    """
    Load, align, and return a clean price panel for *tickers* on *calendar*.

    Parameters
    ----------
    tickers : list[str] or None
        Bloomberg tickers to load.  Defaults to ALL_BLOOMBERG_TICKERS.
    calendar : DatetimeIndex or None
        If None, built from *start* / *end*.
    start, end : str
        Ignored if *calendar* is provided.

    Returns
    -------
    pd.DataFrame  shape (T, N)  — price levels, daily calendar.
    """
    if tickers is None:
        tickers = ALL_BLOOMBERG_TICKERS
    if calendar is None:
        calendar = build_calendar(start, end)

    raw_panel = load_bloomberg_all(tickers, daily_only=False)
    aligned = {}
    for ticker in tickers:
        if ticker not in raw_panel.columns:
            continue
        aligned[ticker] = align_to_calendar(raw_panel[ticker], calendar, ticker=ticker)

    return pd.DataFrame(aligned, index=calendar)


def build_return_panel(
    tickers: list[str] | None = None,
    calendar: pd.DatetimeIndex | None = None,
    start: str = "1987-01-02",
    end: str | None = None,
) -> pd.DataFrame:
    """
    Return a panel of daily log returns for *tickers*.

    Drops the first row (NaN from diff) and any rows where ALL series are NaN.
    """
    prices = build_price_panel(tickers, calendar, start, end)
    returns = prices.apply(compute_log_returns)
    returns = returns.iloc[1:]  # drop first NaN row
    returns = returns.dropna(how="all")
    return returns


def build_clean_dataset(
    start: str = "1987-01-02",
    end: str | None = None,
) -> dict[str, pd.DataFrame]:
    """
    One-stop function that returns a dict with everything needed downstream:

        "prices"    – aligned price panel (all Bloomberg tickers)
        "returns"   – log-return panel (Bloomberg tickers)
        "ff5"       – FF5 daily factor returns (decimal)
        "rf"        – daily risk-free rate (decimal)
        "calendar"  – the DatetimeIndex used
    """
    calendar = build_calendar(start, end)

    prices  = build_price_panel(calendar=calendar)
    returns = build_return_panel(calendar=calendar)
    ff5     = align_ff5(calendar)
    rf      = build_rf_daily(calendar)

    return {
        "prices":   prices,
        "returns":  returns,
        "ff5":      ff5,
        "rf":       rf,
        "calendar": calendar,
    }
