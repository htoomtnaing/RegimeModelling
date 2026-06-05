import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def compute_log_returns(prices):
    """Compute daily log returns from adjusted prices."""
    returns = np.log(prices / prices.shift(1))
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

    if returns.empty:
        raise ValueError("Return data is empty after removing missing values.")

    return returns


def standardize_returns(returns):
    """Standardize returns for model fitting."""
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(returns)
    scaled_returns = pd.DataFrame(
        scaled_values,
        index=returns.index,
        columns=returns.columns,
    )
    return scaled_returns, scaler


def cumulative_returns(returns):
    """Convert log returns into cumulative return series."""
    return np.exp(returns.cumsum()) - 1
