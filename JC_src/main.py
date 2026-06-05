from src.analysis import compact_interpretation_table, regime_correlations, regime_summary
from src.config import (
    COVARIANCE_TYPE,
    END_DATE,
    N_COMPONENTS,
    RANDOM_STATE,
    START_DATE,
    TICKERS,
)
from src.data import download_adjusted_prices
from src.features import compute_log_returns, cumulative_returns, standardize_returns
from src.gmm_model import fit_gmm_regimes, validate_regime_outputs
from src.plots import (
    plot_factor_volatility_table,
    plot_factor_return_table,
    plot_cumulative_returns,
    plot_regime_colored_series,
    plot_regime_correlation_heatmaps,
    plot_regime_probabilities,
)


def run_pipeline():
    """Run the full market regime detection workflow."""
    print("Downloading adjusted close prices...")
    prices = download_adjusted_prices(TICKERS, START_DATE, END_DATE)

    print("Computing log returns and standardized features...")
    returns = compute_log_returns(prices)
    scaled_returns, _ = standardize_returns(returns)

    print("Fitting 4-regime Gaussian Mixture Model...")
    _, labels, probabilities = fit_gmm_regimes(
        scaled_returns,
        n_components=N_COMPONENTS,
        covariance_type=COVARIANCE_TYPE,
        random_state=RANDOM_STATE,
    )
    validate_regime_outputs(returns, labels, probabilities, N_COMPONENTS)

    print("Analyzing regimes...")
    summary = regime_summary(returns, labels)
    interpretation = compact_interpretation_table(summary)
    correlations = regime_correlations(returns, labels)

    print("\nRegime interpretation table:")
    print(interpretation.round(4))

    print("\nCreating figures...")
    plot_cumulative_returns(returns)
    plot_regime_colored_series(
        cumulative_returns(returns),
        labels,
        column="Equity",
    )
    plot_regime_probabilities(probabilities)
    plot_regime_correlation_heatmaps(correlations)
    plot_factor_volatility_table(summary)
    plot_factor_return_table(summary)

    print("\nDone.")
    return prices, returns, labels, probabilities, summary


if __name__ == "__main__":
    run_pipeline()
