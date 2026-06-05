import pandas as pd
from sklearn.mixture import GaussianMixture


def fit_gmm_regimes(scaled_returns, n_components=4, covariance_type="full", random_state=42):
    """Fit a GMM and return the model, regime labels, and regime probabilities."""
    model = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state,
    )
    model.fit(scaled_returns)

    labels = pd.Series(
        model.predict(scaled_returns),
        index=scaled_returns.index,
        name="Regime",
    )

    probabilities = pd.DataFrame(
        model.predict_proba(scaled_returns),
        index=scaled_returns.index,
        columns=[f"Regime {i}" for i in range(n_components)],
    )

    return model, labels, probabilities


def gmm_model_selection(scaled_returns, component_range, covariance_type="full", random_state=42):
    """Fit GMMs over component counts and return AIC/BIC diagnostics."""
    rows = []
    for n_components in component_range:
        model = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=random_state,
        )
        model.fit(scaled_returns)
        rows.append(
            {
                "n_components": n_components,
                "AIC": model.aic(scaled_returns),
                "BIC": model.bic(scaled_returns),
            }
        )
    return pd.DataFrame(rows).set_index("n_components")


def validate_regime_outputs(returns, labels, probabilities, n_components):
    """Run basic sanity checks on model outputs."""
    if not labels.index.equals(returns.index):
        raise ValueError("Regime labels do not align with returns index.")

    if not probabilities.index.equals(returns.index):
        raise ValueError("Regime probabilities do not align with returns index.")

    probability_sums = probabilities.sum(axis=1)
    if not probability_sums.round(6).eq(1.0).all():
        raise ValueError("Regime probabilities do not sum to 1.0 for every row.")

    observed = set(labels.unique())
    expected = set(range(n_components))
    missing = sorted(expected - observed)
    if missing:
        print(f"Warning: these regimes have no assigned observations: {missing}")
