import numpy as np
import pandas as pd


def regime_summary(returns, labels):
    """Summarize average return, volatility, Sharpe, frequency, and average correlation."""
    rows = []
    total_days = len(returns)

    for regime in sorted(labels.unique()):
        regime_returns = returns.loc[labels == regime]
        annual_return = regime_returns.mean() * 252
        annual_volatility = regime_returns.std() * np.sqrt(252)
        annual_sharpe = annual_return / annual_volatility.replace(0, np.nan)
        corr = regime_returns.corr()

        rows.append(
            {
                "Regime": regime,
                "Frequency": len(regime_returns) / total_days,
                "AvgCorrelation": average_correlation(corr),
                **{f"{col}_Return": annual_return[col] for col in returns.columns},
                **{f"{col}_Volatility": annual_volatility[col] for col in returns.columns},
                **{f"{col}_Sharpe": annual_sharpe[col] for col in returns.columns},
            }
        )

    return pd.DataFrame(rows).set_index("Regime")


def average_correlation(correlation_matrix):
    """Average off-diagonal correlation for one regime."""
    if correlation_matrix.shape[0] < 2:
        return np.nan

    mask = ~np.eye(correlation_matrix.shape[0], dtype=bool)
    return correlation_matrix.where(mask).stack().mean()


def regime_correlations(returns, labels):
    """Calculate one correlation matrix per regime."""
    correlations = {}
    for regime in sorted(labels.unique()):
        correlations[regime] = returns.loc[labels == regime].corr()
    return correlations


def transition_matrix(labels, n_regimes=None, normalize=False):
    """Count or normalize one-step transitions between regime labels."""
    labels = pd.Series(labels).dropna().astype(int)
    if n_regimes is None:
        regimes = sorted(labels.unique())
    else:
        regimes = list(range(n_regimes))

    matrix = pd.DataFrame(0.0, index=regimes, columns=regimes)
    for current, following in zip(labels.iloc[:-1], labels.iloc[1:]):
        matrix.loc[current, following] += 1

    if normalize:
        row_sums = matrix.sum(axis=1).replace(0, np.nan)
        matrix = matrix.div(row_sums, axis=0).fillna(0.0)

    matrix.index.name = "From"
    matrix.columns.name = "To"
    return matrix


def regime_duration_stats(labels):
    """Summarize consecutive run lengths for each regime."""
    labels = pd.Series(labels).dropna().astype(int)
    if labels.empty:
        return pd.DataFrame()

    run_id = labels.ne(labels.shift()).cumsum()
    runs = pd.DataFrame(
        {
            "Regime": labels.groupby(run_id).first().to_numpy(),
            "Duration": labels.groupby(run_id).size().to_numpy(),
        }
    )

    stats = runs.groupby("Regime")["Duration"].agg(
        Count="count",
        Mean="mean",
        Median="median",
        Min="min",
        Max="max",
    )
    return stats


def regime_performance_summary(returns, labels, annualization=252):
    """Return annualized return, volatility, and Sharpe for each factor by regime."""
    rows = []
    for regime in sorted(pd.Series(labels).dropna().unique()):
        regime_returns = returns.loc[labels == regime]
        annual_return = regime_returns.mean() * annualization
        annual_volatility = regime_returns.std() * np.sqrt(annualization)
        sharpe = annual_return / annual_volatility.replace(0, np.nan)
        for column in returns.columns:
            rows.append(
                {
                    "Regime": regime,
                    "Factor": column,
                    "AnnualReturn": annual_return[column],
                    "AnnualVolatility": annual_volatility[column],
                    "Sharpe": sharpe[column],
                    "Observations": len(regime_returns),
                }
            )
    return pd.DataFrame(rows)


def label_confusion_matrix(left_labels, right_labels, left_name="GMM", right_name="HMM"):
    """Compare two regime label assignments on the same index."""
    if not left_labels.index.equals(right_labels.index):
        raise ValueError("Label indices must match before building a confusion matrix.")

    matrix = pd.crosstab(left_labels, right_labels)
    matrix.index.name = left_name
    matrix.columns.name = right_name
    return matrix


def persistence_table(transition_probabilities):
    """Extract stay probabilities from a transition probability matrix."""
    rows = []
    for regime in transition_probabilities.index:
        rows.append({"Regime": regime, "Persistence": transition_probabilities.loc[regime, regime]})
    return pd.DataFrame(rows).set_index("Regime")


def compact_interpretation_table(summary):
    """Select the most useful fields for economic interpretation."""
    preferred_columns = [
        "Frequency",
        "AvgCorrelation",
        "Equity_Return",
        "EM_Return",
        "Rates_Return",
        "Credit_Return",
        "Commodities_Return",
        "FX_Return",
        "Inflation_Return",
        "Equity_Volatility",
        "Equity_Sharpe",
    ]
    columns = [col for col in preferred_columns if col in summary.columns]
    return summary[columns].copy()
