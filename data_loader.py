"""
data_loader.py
==============
Loads all raw data sources into clean pandas Series / DataFrames.

Each public function returns a pd.Series with a DatetimeIndex (name = ticker).
The loader handles:
  - Bloomberg xlsx files whose Date column may be either an already-parsed
    datetime or an Excel serial integer.
  - The mixed monthly→daily frequency transitions found in several series
    (MXCXDMHR, LGY7TRUH, LUACTRUU, LF98TRUU, LBUSTRUU, LEGATRUU, EMUSTRUU).
  - Fama-French 5-factor CSV (4-row header, YYYYMMDD integer index, % units).
  - CPI / interest-rate macro CSVs.
  - G10 FX spot rates (Currency_Prices.xlsx, Bloomberg BGN quotes).
  - IMF/World Bank annual GDP by country (Country_GDP_in_USD.csv).

No returns are computed here — that is done in data_cleaner.py.
"""

from __future__ import annotations
import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

# ── Path configuration ──────────────────────────────────────────────────────

# Resolve DATA_ROOT relative to this file so the project works regardless of
# the current working directory.
_HERE = Path(__file__).resolve().parent
DATA_ROOT = _HERE / "data"

BLOOMBERG_DIR = DATA_ROOT / "Bloomberg_Data"
MACRO_DIR     = DATA_ROOT / "Macro"
FF_DIR        = DATA_ROOT / "Farma_French"

# ── Known frequency-transition dates ─────────────────────────────────────────
# Before these dates the Bloomberg series were published monthly; after them
# they are daily.  Used by load_bloomberg() to tag / filter if needed.
DAILY_FROM: dict[str, str] = {
    "MXCXDMHR": "2002-01-02",
    "LGY7TRUH":  "2001-11-22",
    "LUACTRUU":  "1989-01-03",
    "LF98TRUU":  "1998-08-07",
    "LBUSTRUU":  "1989-01-03",
    "LEGATRUU":  "1999-01-04",
    "EMUSTRUU":  "1997-01-02",
    "LP05TRUH":  "2000-08-17",  # Euro IG hedged: monthly 1999, daily from Aug-2000
    "LP01TRUH":  "2000-07-07",  # Euro HY hedged: monthly 1999, daily from Jul-2000
    # Everything else was daily from inception in our data
}


# ── Internal helpers ─────────────────────────────────────────────────────────

def _parse_bloomberg_dates(raw: pd.Series) -> pd.Series:
    """
    Bloomberg xlsx Date columns come in two flavours:
      1. Already a datetime64 (openpyxl parsed it).
      2. A float/int Excel serial number (days since 1899-12-30).
    Returns a Series of Timestamp (NaT for any unparseable value).
    """
    if pd.api.types.is_datetime64_any_dtype(raw):
        return pd.to_datetime(raw, errors="coerce")
    # Numeric serial
    try:
        return pd.to_datetime(raw, origin="1899-12-30", unit="D", errors="coerce")
    except Exception:
        # Fallback: try generic parse
        return pd.to_datetime(raw, errors="coerce")


def _read_xlsx(path: Path, sheet: int | str = 0) -> pd.DataFrame:
    """Read an xlsx file; return raw two-column DataFrame [Date, Price]."""
    df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
    if df.shape[1] < 2:
        raise ValueError(f"{path.name}: expected at least 2 columns, got {df.shape[1]}")
    df = df.iloc[:, :2].copy()
    df.columns = ["Date", "Price"]
    df["Date"] = _parse_bloomberg_dates(df["Date"])
    df = (
        df.dropna(subset=["Date"])
          .sort_values("Date")
          .drop_duplicates(subset="Date", keep="last")
          .reset_index(drop=True)
    )
    return df


# ── Public loaders ────────────────────────────────────────────────────────────

def load_bloomberg(ticker: str, daily_only: bool = False) -> pd.Series:
    """
    Load a Bloomberg index by ticker string.

    Parameters
    ----------
    ticker : str
        Must match a file name pattern ``{ticker}_*.xlsx`` in BLOOMBERG_DIR.
    daily_only : bool
        If True, drop all rows before the known daily-from date for series
        that started as monthly.  Safe to leave False — data_cleaner.py
        forward-fills on a daily calendar anyway, so monthly obs before the
        transition become the carry-forward value until the next monthly point.

    Returns
    -------
    pd.Series
        Price levels indexed by date, named ``ticker``.
    """
    matches = sorted(BLOOMBERG_DIR.glob(f"{ticker}_*.xlsx"))
    if not matches:
        raise FileNotFoundError(
            f"No Bloomberg file found for ticker '{ticker}' in {BLOOMBERG_DIR}"
        )
    path = matches[0]
    df = _read_xlsx(path)

    if daily_only and ticker in DAILY_FROM:
        cutoff = pd.Timestamp(DAILY_FROM[ticker])
        df = df[df["Date"] >= cutoff].reset_index(drop=True)

    s = df.set_index("Date")["Price"].rename(ticker)
    return s


def load_bloomberg_all(tickers: list[str], daily_only: bool = False) -> pd.DataFrame:
    """Load multiple Bloomberg tickers and return as a wide DataFrame."""
    series = [load_bloomberg(t, daily_only=daily_only) for t in tickers]
    return pd.concat(series, axis=1)


def load_ff5_daily() -> pd.DataFrame:
    """
    Load the Fama-French 5-factor daily file.

    Returns a DataFrame with columns:
        Mkt_RF, SMB, HML, RMW, CMA, RF
    Values are in decimal (divided by 100 from the raw percent file).
    """
    matches = sorted(FF_DIR.glob("F-F_Research_Data_5_Factors_2x3_US_daily*.csv"))
    if not matches:
        raise FileNotFoundError(f"FF5 daily CSV not found in {FF_DIR}")
    path = matches[0]

    # The file has a 3-row descriptive header before the data header row
    df = pd.read_csv(path, skiprows=3)
    df.columns = ["Date", "Mkt_RF", "SMB", "HML", "RMW", "CMA", "RF"]
    df["Date"] = pd.to_datetime(df["Date"].astype(str), format="%Y%m%d", errors="coerce")
    df = (
        df.dropna(subset=["Date"])
          .sort_values("Date")
          .drop_duplicates(subset="Date", keep="last")
          .reset_index(drop=True)
          .set_index("Date")
    )
    # Convert from percent to decimal
    df = df / 100.0
    return df


def load_ff5_monthly() -> pd.DataFrame:
    """
    Load the Fama-French 5-factor monthly file.
    Values returned in decimal.
    """
    matches = sorted(FF_DIR.glob("F-F_Research_Data_5_Factors_2x3_US_monthly*.csv"))
    if not matches:
        raise FileNotFoundError(f"FF5 monthly CSV not found in {FF_DIR}")
    path = matches[0]
    df = pd.read_csv(path, skiprows=3)
    df.columns = ["Date", "Mkt_RF", "SMB", "HML", "RMW", "CMA", "RF"]
    df["Date"] = pd.to_datetime(df["Date"].astype(str) + "01", format="%Y%m%d", errors="coerce")
    # Shift to month-end
    df["Date"] = df["Date"] + pd.offsets.MonthEnd(0)
    df = (
        df.dropna(subset=["Date"])
          .sort_values("Date")
          .drop_duplicates(subset="Date", keep="last")
          .reset_index(drop=True)
          .set_index("Date")
    )
    df = df / 100.0
    return df


def load_cpi_interest_rates() -> pd.DataFrame:
    """
    Load the CPI / interest-rate quarterly panel (1925-12 to 2024-12).

    Columns of interest:
        cpiret   – quarterly CPI return (decimal after /100 NOT needed; already decimal)
        cpiind   – CPI index level
        t30ret   – 30-day T-bill return
        t90ret   – 90-day T-bill return
        b10ret   – 10-year bond return
        b10ind   – 10-year bond index
    """
    matches = sorted(MACRO_DIR.glob("CPI_InterestRates*.csv"))
    if not matches:
        raise FileNotFoundError(f"CPI_InterestRates CSV not found in {MACRO_DIR}")
    path = matches[0]
    df = pd.read_csv(path)
    df["caldt"] = pd.to_datetime(df["caldt"], errors="coerce")
    df = (
        df.dropna(subset=["caldt"])
          .sort_values("caldt")
          .drop_duplicates(subset="caldt", keep="last")
          .set_index("caldt")
    )
    return df


def load_interest_rate_daily() -> pd.DataFrame:
    """
    Load the daily interest-rate panel (FRED + others, 1954-01 to 2025-02).

    Key columns surfaced for downstream use:
        dff    – Fed Funds Rate (most complete, 1954-)
        effr   – Effective Fed Funds Rate (2000-)
        dtb3   – 3-month T-bill (1954-)
        dgs3mo – 3-month Treasury constant maturity (1981-)
        sofr   – SOFR (2023-)

    The preferred risk-free proxy is constructed in data_cleaner.py by
    stitching dff → dtb3 → dgs3mo depending on availability.
    """
    matches = sorted(MACRO_DIR.glob("Interest_Rate_Daily*.csv"))
    if not matches:
        raise FileNotFoundError(f"Interest_Rate_Daily CSV not found in {MACRO_DIR}")
    path = matches[0]
    df = pd.read_csv(path, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = (
        df.dropna(subset=["date"])
          .sort_values("date")
          .drop_duplicates(subset="date", keep="last")
          .set_index("date")
    )
    # Convert any object columns that should be numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ── G10 FX spot rates ─────────────────────────────────────────────────────────

# Bloomberg BGN column names to short ticker mapping
_FX_COL_MAP: dict[str, str] = {
    "EURUSD BGN Curncy": "EURUSD",
    "USDJPY BGN Curncy": "USDJPY",
    "USDCHF BGN Curncy": "USDCHF",
    "USDCAD BGN Curncy": "USDCAD",
    "AUDUSD BGN Curncy": "AUDUSD",
    "NZDUSD BGN Curncy": "NZDUSD",
    "USDNOK BGN Curncy": "USDNOK",
    "USDSEK BGN Curncy": "USDSEK",
    "GBPUSD BGN Curncy": "GBPUSD",
}

# Pairs quoted as foreign-per-USD (must invert to get USD-per-foreign log returns)
FX_INVERT: set[str] = {"USDJPY", "USDCHF", "USDCAD", "USDNOK", "USDSEK"}

# Pairs already quoted as USD-per-foreign (use log return directly)
FX_DIRECT: set[str] = {"EURUSD", "AUDUSD", "NZDUSD", "GBPUSD"}


def load_fx_prices() -> pd.DataFrame:
    """
    Load G10 FX spot prices from Currency_Prices.xlsx (Bloomberg BGN quotes).

    Returns a DataFrame of spot price levels (not returns) with columns renamed
    to short tickers (EURUSD, USDJPY, etc.).  Prices are in native Bloomberg
    convention:
        USD-per-foreign : EURUSD, AUDUSD, NZDUSD, GBPUSD
        Foreign-per-USD : USDJPY, USDCHF, USDCAD, USDNOK, USDSEK

    The inversion to a uniform USD-per-foreign basis is applied in
    secondary_macro.build_foreign_currency() when computing log returns.

    Notes
    -----
    - EUR before 1999 is a Bloomberg BGN synthetic (DEM/ECU-based) continuous
      series with no artificial discontinuity at the 1999-01-04 EUR launch.
    - Repeated prices (weekend/holiday carries) are handled by the business-day
      calendar reindex + ffill in data_cleaner.py.
    """
    matches = sorted(BLOOMBERG_DIR.glob("Currency_Prices*.xlsx"))
    if not matches:
        raise FileNotFoundError(
            f"Currency_Prices.xlsx not found in {BLOOMBERG_DIR}"
        )
    path = matches[0]
    df = pd.read_excel(path, engine="openpyxl")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = (
        df.dropna(subset=["Date"])
          .sort_values("Date")
          .drop_duplicates(subset="Date", keep="last")
          .set_index("Date")
    )
    df = df.rename(columns=_FX_COL_MAP)
    keep = [c for c in _FX_COL_MAP.values() if c in df.columns]
    return df[keep].apply(pd.to_numeric, errors="coerce")


# ── GDP data ──────────────────────────────────────────────────────────────────

# Map from FX ticker to GDP-country list that proxies the currency economy.
# EUR uses France + Germany + Italy + Spain (Netherlands absent from our data;
# these four cover approximately 85 pct of the four-country group GDP).
FX_GDP_MAP: dict[str, list[str]] = {
    "EURUSD": ["France", "Germany", "Italy", "Spain"],
    "USDJPY": ["Japan"],
    "USDCHF": ["Switzerland"],
    "USDCAD": ["Canada"],
    "AUDUSD": ["Australia"],
    "NZDUSD": ["New Zealand"],
    "USDNOK": ["Norway"],
    "USDSEK": ["Sweden"],
    "GBPUSD": ["United Kingdom"],
}


def load_gdp() -> pd.DataFrame:
    """
    Load annual nominal GDP in USD billions by country.

    Returns a DataFrame indexed by integer Year with one column per country.
    Covers 1980-2026 (IMF WEO; 2025-2026 are IMF forecasts).
    """
    matches = sorted(MACRO_DIR.glob("Country_GDP_in_USD*.csv"))
    if not matches:
        raise FileNotFoundError(
            f"Country_GDP_in_USD CSV not found in {MACRO_DIR}"
        )
    path = matches[0]
    df = pd.read_csv(path).set_index("Year")
    return df.apply(pd.to_numeric, errors="coerce")


# ── Convenience: load everything at once ─────────────────────────────────────

# All Bloomberg tickers we have on disk
ALL_BLOOMBERG_TICKERS = [
    "MXCXDMHR",  # MSCI ACWI hedged USD  → Equity base
    "LGY7TRUH",  # Global Govt 7-10yr hedged → Rates base
    "LUACTRUU",  # US Corporate IG          → Credit (US IG leg)
    "LF98TRUU",  # US Corporate HY          → Credit (US HY leg)
    "LP05TRUH",  # Pan Euro Agg Corp hedged → Credit (EU IG leg)
    "LP01TRUH",  # Pan Euro HY hedged       → Credit (EU HY leg)
    "BCOMTR",    # Bloomberg Commodity TR    → Commodities base
    "MXEF",      # MSCI EM                  → EM Equity
    "EMUSTRUU",  # Barclays EM USD          → EM Credit
    "BCIT5T",    # US TIPS 7-10yr           → Local Inflation
    "PUT",       # CBOE PutWrite            → Short Volatility
    "MXWD",      # MSCI ACWI unhedged       → EM relative / reference
    "LEGATRUU",  # Global Agg               → reference
    "RU30INTR",  # Russell 3000 TR          → Local Equity / reference
    "SPGSCITR",  # S&P GSCI TR              → alternative commodities
    "LBUSTRUU",  # US Agg Bond              → rates reference
    "JPEIGLBL",  # JP Morgan EMBI           → EM bonds alternative
    "M1EF",      # MSCI EM Net Return       → EM equity alternative
]


def load_all_bloomberg(daily_only: bool = False) -> pd.DataFrame:
    """Load all available Bloomberg tickers into a single wide DataFrame."""
    return load_bloomberg_all(ALL_BLOOMBERG_TICKERS, daily_only=daily_only)
