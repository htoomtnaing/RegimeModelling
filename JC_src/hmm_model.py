import pandas as pd
from hmmlearn.hmm import GaussianHMM


def fit_hmm_regimes(
    scaled_returns,
    n_components=4,
    covariance_type="full",
    random_state=42,
    n_iter=1000,
):
    """Fit a Gaussian HMM and return the model, hidden states, and state probabilities."""
    model = GaussianHMM(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state,
        n_iter=n_iter,
    )
    model.fit(scaled_returns)

    labels = pd.Series(
        model.predict(scaled_returns),
        index=scaled_returns.index,
        name="HiddenState",
    )

    probabilities = pd.DataFrame(
        model.predict_proba(scaled_returns),
        index=scaled_returns.index,
        columns=[f"State {i}" for i in range(n_components)],
    )

    return model, labels, probabilities


def validate_hmm_outputs(returns, labels, probabilities, n_components):
    """Run basic sanity checks on HMM outputs."""
    if not labels.index.equals(returns.index):
        raise ValueError("HMM labels do not align with returns index.")

    if not probabilities.index.equals(returns.index):
        raise ValueError("HMM probabilities do not align with returns index.")

    probability_sums = probabilities.sum(axis=1)
    if not probability_sums.round(6).eq(1.0).all():
        raise ValueError("HMM probabilities do not sum to 1.0 for every row.")

    observed = set(labels.unique())
    expected = set(range(n_components))
    missing = sorted(expected - observed)
    if missing:
        print(f"Warning: these hidden states have no assigned observations: {missing}")
