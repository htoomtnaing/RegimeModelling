import pandas as pd
import yfinance as yf


def download_adjusted_prices(tickers, start_date, end_date=None):
    """Download adjusted close prices and return columns named by factor."""
    ticker_list = list(tickers.values())
    raw = yf.download(
        ticker_list,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
    )

    if raw.empty:
        raise ValueError("No price data was downloaded. Check tickers or internet access.")

    if "Adj Close" in raw.columns:
        prices = raw["Adj Close"].copy()
    elif "Close" in raw.columns:
        prices = raw["Close"].copy()
    else:
        raise ValueError("Downloaded data did not contain adjusted close or close prices.")

    reverse_names = {ticker: name for name, ticker in tickers.items()}
    prices = prices.rename(columns=reverse_names)
    prices = prices[list(tickers.keys())]
    prices = prices.dropna(how="all")
    prices = prices.dropna(axis=1, how="all")
    prices = prices.ffill().dropna()

    missing = sorted(set(tickers.keys()) - set(prices.columns))
    if missing:
        raise ValueError(f"Missing price columns after download: {missing}")

    return prices
