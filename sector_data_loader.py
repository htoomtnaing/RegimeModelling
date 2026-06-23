"""
sector_data_loader.py
=====================
Downloads and caches SPDR Select Sector ETF price data from Yahoo Finance.

Tickers
-------
XLK   Technology
XLF   Financials
XLE   Energy
XLV   Health Care
XLI   Industrials
XLY   Consumer Discretionary
XLP   Consumer Staples
XLU   Utilities
XLRE  Real Estate          (inception 2015-10-07; filled with VNQ pre-2015)
XLB   Materials
XLC   Communication Svcs   (inception 2018-06-18; filled with VOX pre-2018)

All ETFs are total-return (dividends reinvested via Yahoo Finance adjusted close).
Data is cached locally as CSV to avoid repeated downloads.

Usage
-----
    from sector_data_loader import load_sector_returns

    returns = load_sector_returns(start='2002-01-02', end=None, cache=True)
    # Returns a DataFrame of daily log-returns, index=Date, columns=tickers
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── Sector ETF universe ───────────────────────────────────────────────────────

SECTOR_TICKERS = [
    "XLK",   # Technology
    "XLF",   # Financials
    "XLE",   # Energy
    "XLV",   # Health Care
    "XLI",   # Industrials
    "XLY",   # Consumer Discretionary
    "XLP",   # Consumer Staples
    "XLU",   # Utilities
    "XLRE",  # Real Estate
    "XLB",   # Materials
    "XLC",   # Communication Services
]

SECTOR_LABELS = {
    "XLK":  "Technology",
    "XLF":  "Financials",
    "XLE":  "Energy",
    "XLV":  "Health Care",
    "XLI":  "Industrials",
    "XLY":  "Consumer Discretionary",
    "XLP":  "Consumer Staples",
    "XLU":  "Utilities",
    "XLRE": "Real Estate",
    "XLB":  "Materials",
    "XLC":  "Communication Services",
}

# Proxies for tickers with short histories
# XLRE launched 2015-10-07 → backfill with VNQ (Vanguard Real Estate ETF, from 2004)
# XLC  launched 2018-06-18 → backfill with VOX (Vanguard Comm Svcs ETF, from 2004)
_BACKFILL_PROXIES = {
    "XLRE": "VNQ",
    "XLC":  "VOX",
}

# Cache directory (sits alongside this file)
_CACHE_DIR = Path(__file__).parent / "Data" / "Sector_ETF"


# ── Main loader ───────────────────────────────────────────────────────────────

def load_sector_returns(
    start:       str  = "2002-01-02",
    end:         Optional[str] = None,
    cache:       bool = True,
    cache_dir:   Optional[Path] = None,
    log_returns: bool = True,
    verbose:     bool = True,
) -> pd.DataFrame:
    """
    Download (or load from cache) adjusted-close prices for all sector ETFs
    and return a DataFrame of daily returns.

    Parameters
    ----------
    start       : start date string "YYYY-MM-DD"
    end         : end date string or None (= today)
    cache       : if True, save/load prices from CSV cache
    cache_dir   : override default cache directory
    log_returns : if True return log-returns, else simple returns
    verbose     : print progress

    Returns
    -------
    pd.DataFrame  shape (T, 11), index=DatetimeIndex, columns=SECTOR_TICKERS
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError(
            "yfinance is not installed. Run:  pip install yfinance"
        )

    cache_dir = Path(cache_dir) if cache_dir else _CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"sector_prices_{start[:10].replace('-','')}.csv"

    # ── Load from cache ───────────────────────────────────────────────────────
    if cache and cache_file.exists():
        if verbose:
            print(f"[sector_data_loader] Loading cached prices from {cache_file}")
        prices = pd.read_csv(cache_file, index_col="Date", parse_dates=True)
        # Refresh if stale (last row older than 2 business days)
        last = prices.index[-1]
        today = pd.Timestamp.today().normalize()
        stale = (today - last).days > 4   # allow for weekends
        if not stale:
            return _prices_to_returns(prices, log_returns, verbose)
        if verbose:
            print(f"  Cache stale (last={last.date()}), refreshing...")

    # ── Download from Yahoo Finance ───────────────────────────────────────────
    all_tickers = SECTOR_TICKERS + list(set(_BACKFILL_PROXIES.values()))
    if verbose:
        print(f"[sector_data_loader] Downloading {len(all_tickers)} tickers from Yahoo Finance...")

    raw = yf.download(
        all_tickers,
        start=start,
        end=end,
        auto_adjust=True,   # adjusted close (dividends + splits)
        progress=False,
        threads=True,
    )

    # yfinance returns MultiIndex columns (metric, ticker) when >1 ticker
    if isinstance(raw.columns, pd.MultiIndex):
        prices_raw = raw["Close"]
    else:
        prices_raw = raw[["Close"]].rename(columns={"Close": all_tickers[0]})

    prices_raw.index = pd.to_datetime(prices_raw.index)
    prices_raw.index.name = "Date"

    # ── Backfill short-history tickers ────────────────────────────────────────
    prices = pd.DataFrame(index=prices_raw.index)
    for ticker in SECTOR_TICKERS:
        if ticker in prices_raw.columns:
            col = prices_raw[ticker].copy()
            if ticker in _BACKFILL_PROXIES:
                proxy = _BACKFILL_PROXIES[ticker]
                # Find ETF inception date
                first_valid = col.first_valid_index()
                if first_valid is not None and proxy in prices_raw.columns:
                    proxy_col = prices_raw[proxy]
                    # Scale proxy so its level matches ETF at inception
                    if first_valid in col.index and first_valid in proxy_col.index:
                        scale = col.loc[first_valid] / proxy_col.loc[first_valid]
                        backfill = proxy_col.loc[:first_valid] * scale
                        col = pd.concat([backfill.iloc[:-1], col])
                        col = col[~col.index.duplicated(keep='last')]
                        if verbose:
                            print(f"  {ticker}: backfilled with {proxy} before {first_valid.date()}")
            prices[ticker] = col
        else:
            if verbose:
                print(f"  WARNING: {ticker} not found in download — filling with NaN")
            prices[ticker] = np.nan

    # Restrict to requested date range
    prices = prices.loc[start:]
    if end:
        prices = prices.loc[:end]

    # Forward-fill up to 3 days (handles market holidays)
    prices = prices.ffill(limit=3)

    # ── Save cache ────────────────────────────────────────────────────────────
    if cache:
        prices.to_csv(cache_file)
        if verbose:
            print(f"  Cached to {cache_file}")

    return _prices_to_returns(prices, log_returns, verbose)


def _prices_to_returns(
    prices:      pd.DataFrame,
    log_returns: bool = True,
    verbose:     bool = True,
) -> pd.DataFrame:
    """Convert price DataFrame to return DataFrame."""
    if log_returns:
        rets = np.log(prices / prices.shift(1)).iloc[1:]
    else:
        rets = prices.pct_change().iloc[1:]

    # Drop rows where all values are NaN (e.g. first row or non-trading days)
    rets = rets.dropna(how="all")

    if verbose:
        print(f"  Returns shape: {rets.shape}  "
              f"({rets.index[0].date()} → {rets.index[-1].date()})")
        missing = rets.isna().sum()
        if missing.any():
            print(f"  NaN counts: {missing[missing>0].to_dict()}")

    return rets[SECTOR_TICKERS]


# ── Convenience: load prices (not returns) ────────────────────────────────────

def load_sector_prices(
    start:     str  = "2002-01-02",
    end:       Optional[str] = None,
    cache:     bool = True,
    verbose:   bool = True,
) -> pd.DataFrame:
    """Return adjusted-close price levels instead of returns."""
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("pip install yfinance")

    cache_dir = _CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"sector_prices_{start[:10].replace('-','')}.csv"

    if cache and cache_file.exists():
        prices = pd.read_csv(cache_file, index_col="Date", parse_dates=True)
        today  = pd.Timestamp.today().normalize()
        if (today - prices.index[-1]).days <= 4:
            return prices[SECTOR_TICKERS]

    # Re-use the loader (it caches prices before converting to returns)
    load_sector_returns(start=start, end=end, cache=cache, verbose=verbose)
    prices = pd.read_csv(cache_file, index_col="Date", parse_dates=True)
    return prices[SECTOR_TICKERS]


# ── Diagnostic ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Downloading sector ETF data...")
    rets = load_sector_returns(start="2002-01-02", verbose=True)
    print("\nAnnualised returns:")
    ann = (rets.mean() * 252 * 100).round(2)
    print(ann.to_string())
    print("\nAnnualised vols:")
    vols = (rets.std() * np.sqrt(252) * 100).round(2)
    print(vols.to_string())
    print("\nCorrelation matrix:")
    print(rets.corr().round(2).to_string())
