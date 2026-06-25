"""
regime_utils.py
===============
Analysis, visualisation, and utility functions for the regime model output.

Functions
---------
plot_regime_timeline        – colour-coded horizontal bar of regime history
plot_regime_probabilities   – stacked area chart of probability over time
plot_factor_heatmap         – factor mean returns heatmap across regimes
plot_cv_scores              – log-likelihood vs n_components
compute_transition_matrix   – empirical regime-transition probabilities
compute_regime_stats        – mean/vol/Sharpe per factor per regime
relabel_regimes             – manually override auto-assigned labels
get_regime_periods          – extract start/end dates for each regime period
rolling_regime_window       – walk-forward re-fitting for stability analysis
"""

from __future__ import annotations

from typing import Optional
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

from src.regime_model import RegimeModel, fit_regime_model


# ── Colour scheme (matches Two Sigma paper roughly) ──────────────────────────

REGIME_COLORS = {
    "Steady_State": "#4C9BE8",   # blue
    "WOI":          "#F5C518",   # amber
    "Crisis":       "#E84040",   # red
    "Inflation":    "#F5A623",   # orange
}
DEFAULT_COLORS = ["#4C9BE8", "#F5C518", "#E84040", "#F5A623",
                  "#7ED321", "#9B59B6", "#1ABC9C"]


def _regime_color(label: str, idx: int) -> str:
    return REGIME_COLORS.get(label, DEFAULT_COLORS[idx % len(DEFAULT_COLORS)])


# ── Transition matrix ─────────────────────────────────────────────────────────

def compute_transition_matrix(
    hard_labels: pd.Series,
    regime_names: dict[int, str],
) -> pd.DataFrame:
    """
    Compute the empirical transition probability matrix.

    T[i,j] = P(next regime = j | current regime = i)

    Returns a DataFrame with named rows/columns.
    """
    n = max(regime_names) + 1
    counts = np.zeros((n, n), dtype=float)
    labels = hard_labels.values
    for t in range(len(labels) - 1):
        counts[labels[t], labels[t + 1]] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    prob_matrix = counts / row_sums

    names = [regime_names.get(i, f"Regime_{i}") for i in range(n)]
    return pd.DataFrame(prob_matrix, index=names, columns=names)


# ── Regime statistics ─────────────────────────────────────────────────────────

def compute_regime_stats(
    factor_matrix: pd.DataFrame,
    hard_labels: pd.Series,
    regime_names: dict[int, str],
    trading_days: int = 252,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute annualised mean returns and volatilities per regime.

    Aligns factor_matrix and hard_labels on their common index before
    computing statistics, so mismatched lengths (e.g. when dropna removes
    different rows) are handled cleanly.

    Returns
    -------
    (means_df, vols_df)  — both shaped (n_regimes, n_factors), index = regime labels
    """
    # Align on common index — factor_matrix may have more rows than hard_labels
    # because get_factor_matrix_for_gmm(dropna=True) drops additional NaN rows
    common = factor_matrix.index.intersection(hard_labels.index)
    fm   = factor_matrix.loc[common]
    labs = hard_labels.loc[common]

    means, vols = {}, {}
    for k, name in regime_names.items():
        mask = labs == k
        sub  = fm[mask]
        means[name] = sub.mean() * trading_days
        vols[name]  = sub.std()  * np.sqrt(trading_days)

    return pd.DataFrame(means).T, pd.DataFrame(vols).T


# ── Regime period extraction ──────────────────────────────────────────────────

def get_regime_periods(
    hard_labels: pd.Series,
    regime_names: dict[int, str],
) -> pd.DataFrame:
    """
    Extract a DataFrame of (regime_name, start_date, end_date) tuples
    for each contiguous run of the same regime.

    Useful for overlaying shaded regions on price charts.
    """
    # Map integer labels to names first for cleaner iteration
    named = hard_labels.map(lambda k: regime_names.get(k, f"Regime_{k}"))
    records = []
    prev_label = None
    start_date = None

    for date, name in named.items():
        if name != prev_label:
            if prev_label is not None:
                records.append({"regime": prev_label, "start": start_date, "end": date})
            start_date = date
            prev_label = name

    if prev_label is not None:
        records.append({"regime": prev_label, "start": start_date, "end": named.index[-1]})

    return pd.DataFrame(records)


# ── Plotting functions ────────────────────────────────────────────────────────

def plot_regime_timeline(
    hard_labels: pd.Series,
    regime_names: dict[int, str],
    title: str = "Regime Classification History",
    figsize: tuple = (16, 2.5),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Colour-coded horizontal timeline of regime assignments.
    Each day is coloured by its highest-probability regime.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    dates = hard_labels.index
    labels = hard_labels.values

    for i, (date, label_int) in enumerate(zip(dates, labels)):
        name  = regime_names.get(label_int, f"Regime_{label_int}")
        color = _regime_color(name, label_int)
        ax.axvline(x=i, color=color, linewidth=0.5, alpha=0.9)

    # X-axis: year ticks
    year_idx = [i for i, d in enumerate(dates) if d.month == 1 and d.day <= 7]
    year_labels = [dates[i].year for i in year_idx]
    ax.set_xticks(year_idx)
    ax.set_xticklabels(year_labels, rotation=45, fontsize=8)
    ax.set_yticks([])
    ax.set_xlim(0, len(dates))
    ax.set_title(title, fontsize=12)

    # Legend
    patches = []
    freq = hard_labels.value_counts()
    for k, name in regime_names.items():
        pct = 100 * freq.get(k, 0) / len(hard_labels)
        p = mpatches.Patch(color=_regime_color(name, k), label=f"{name} ({pct:.0f}%)")
        patches.append(p)
    ax.legend(handles=patches, loc="upper left", fontsize=8, ncol=len(regime_names))

    fig.tight_layout()
    return fig


def plot_regime_probabilities(
    probabilities: pd.DataFrame,
    regime_names: dict[int, str],
    title: str = "Regime Probabilities Over Time",
    figsize: tuple = (16, 4),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Stacked area chart of regime probabilities.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    colors = [_regime_color(regime_names.get(k, f"Regime_{k}"), k)
              for k in range(len(probabilities.columns))]
    labels = [regime_names.get(k, f"Regime_{k}") for k in range(len(probabilities.columns))]

    ax.stackplot(
        probabilities.index,
        [probabilities.iloc[:, k].values for k in range(len(probabilities.columns))],
        labels=labels,
        colors=colors,
        alpha=0.85,
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability", fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.legend(loc="upper left", fontsize=8, ncol=len(regime_names))
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    fig.tight_layout()
    return fig


def plot_factor_heatmap(
    means_df: pd.DataFrame,
    vols_df: Optional[pd.DataFrame] = None,
    title: str = "Annualised Factor Mean Returns by Regime (%)",
    figsize: tuple = (14, 6),
) -> plt.Figure:
    """
    Heatmap of annualised factor mean returns across regimes.
    Cells are coloured green (positive) / red (negative).
    Optionally overlays volatility in parentheses.
    """
    data = means_df * 100  # to percent

    fig, ax = plt.subplots(figsize=figsize)

    vmax = max(abs(data.values.max()), abs(data.values.min())) + 1
    im = ax.imshow(data.values, cmap="RdYlGn", aspect="auto",
                   vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels(data.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(data.index, fontsize=10)
    ax.set_title(title, fontsize=12, pad=10)

    for i in range(len(data.index)):
        for j in range(len(data.columns)):
            val = data.values[i, j]
            cell_text = f"{val:.1f}%"
            if vols_df is not None and data.columns[j] in vols_df.columns:
                v = vols_df.values[i, j] * 100
                cell_text += f"\n({v:.1f}%)"
            ax.text(j, i, cell_text, ha="center", va="center",
                    fontsize=8, color="black")

    plt.colorbar(im, ax=ax, shrink=0.8, label="Annualised Return (%)")
    fig.tight_layout()
    return fig


def plot_cv_scores(
    cv_scores: dict[int, float],
    aic_bic: Optional[dict] = None,
    title: str = "Model Selection: CV Log-Likelihood",
    figsize: tuple = (8, 4),
) -> plt.Figure:
    """
    Plot CV log-likelihood (and optionally AIC/BIC) vs number of components.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ns = sorted(cv_scores)
    scores = [cv_scores[n] for n in ns]
    ax.plot(ns, scores, "o-", color="#4C9BE8", label="CV log-likelihood", linewidth=2)
    best_n = max(cv_scores, key=cv_scores.get)
    ax.axvline(best_n, color="#E84040", linestyle="--", alpha=0.7, label=f"Best n={best_n}")

    if aic_bic:
        ax2 = ax.twinx()
        aics = [aic_bic[n]["aic"] for n in ns]
        bics = [aic_bic[n]["bic"] for n in ns]
        ax2.plot(ns, aics, "s--", color="#F5A623", label="AIC", linewidth=1.5)
        ax2.plot(ns, bics, "^--", color="#7ED321", label="BIC", linewidth=1.5)
        ax2.set_ylabel("AIC / BIC", fontsize=9)
        ax2.legend(loc="upper right", fontsize=8)

    ax.set_xlabel("Number of regimes (n_components)", fontsize=10)
    ax.set_ylabel("Mean CV log-likelihood", fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xticks(ns)
    fig.tight_layout()
    return fig


def plot_dashboard(
    model: RegimeModel,
    factor_matrix: pd.DataFrame,
    figsize: tuple = (18, 12),
) -> plt.Figure:
    """
    Four-panel summary dashboard:
        1. Regime timeline
        2. Regime probabilities (stacked area)
        3. Factor mean heatmap
        4. Transition matrix
    """
    fig = plt.figure(figsize=figsize)
    gs  = GridSpec(3, 2, figure=fig, height_ratios=[1, 1.5, 2])

    ax1 = fig.add_subplot(gs[0, :])   # timeline (full width)
    ax2 = fig.add_subplot(gs[1, :])   # probabilities (full width)
    ax3 = fig.add_subplot(gs[2, 0])   # heatmap
    ax4 = fig.add_subplot(gs[2, 1])   # transition matrix

    # 1. Timeline
    plot_regime_timeline(model.hard_labels, model.regime_names, ax=ax1,
                         title="Regime Timeline")

    # 2. Probabilities
    plot_regime_probabilities(model.probabilities, model.regime_names, ax=ax2,
                              title="Regime Probabilities")

    # 3. Factor heatmap
    means_df, vols_df = compute_regime_stats(factor_matrix, model.hard_labels, model.regime_names)
    _plot_heatmap_on_ax(means_df * 100, ax3, title="Factor Means by Regime (%)")

    # 4. Transition matrix
    trans = compute_transition_matrix(model.hard_labels, model.regime_names)
    _plot_heatmap_on_ax(trans * 100, ax4, title="Transition Probabilities (%)",
                        cmap="Blues", fmt=".0f", suffix="%")

    fig.suptitle("Regime Model Dashboard", fontsize=14, y=1.01)
    fig.tight_layout()
    return fig


def _plot_heatmap_on_ax(
    df: pd.DataFrame,
    ax: plt.Axes,
    title: str = "",
    cmap: str = "RdYlGn",
    fmt: str = ".1f",
    suffix: str = "",
) -> None:
    """Internal helper: plot a DataFrame as a heatmap on an existing Axes.

    Parameters
    ----------
    fmt    : Python format spec for the cell value (e.g. ".1f", ".0f")
    suffix : string appended after the formatted value (e.g. "%" or "")
             Use this instead of embedding %% in fmt to avoid f-string conflicts.
    """
    im = ax.imshow(df.values, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index, fontsize=8)
    ax.set_title(title, fontsize=10)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            val = df.values[i, j]
            cell_text = format(val, fmt) + suffix
            ax.text(j, i, cell_text, ha="center", va="center", fontsize=7)


# ── Manual relabelling ────────────────────────────────────────────────────────

def relabel_regimes(
    model: RegimeModel,
    new_labels: dict[int, str],
) -> RegimeModel:
    """
    Overwrite the auto-assigned regime labels with analyst-provided names.

    Parameters
    ----------
    model : RegimeModel
    new_labels : dict {cluster_index: new_name_string}
        E.g. {0: "Crisis", 1: "Steady_State", 2: "WOI", 3: "Inflation"}

    Returns
    -------
    Updated RegimeModel (regime_names field replaced; all other fields unchanged).
    """
    updated = RegimeModel(
        gmm=model.gmm,
        scaler=model.scaler,
        n_components=model.n_components,
        factor_names=model.factor_names,
        probabilities=model.probabilities,
        hard_labels=model.hard_labels,
        regime_names=new_labels,
        factor_means=model.factor_means,
        factor_vols=model.factor_vols,
        cv_scores=model.cv_scores,
    )
    return updated


# ── Walk-forward re-fitting ───────────────────────────────────────────────────

def rolling_regime_window(
    factor_matrix: pd.DataFrame,
    n_components: int = 4,
    train_years: int = 5,
    step_months: int = 3,
    n_init: int = 10,
    random_state: int = 42,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Walk-forward analysis: re-fit the GMM using a rolling training window
    and record the out-of-sample regime probabilities.

    Useful for assessing regime-label stability over time.

    Parameters
    ----------
    factor_matrix : pd.DataFrame  — clean factor return matrix
    n_components : int            — fixed n_components (don't re-run CV each window)
    train_years : int             — years of history in each training window
    step_months : int             — months between re-fittings
    n_init, random_state : int

    Returns
    -------
    pd.DataFrame  shape (T_out, n_components) — out-of-sample probabilities
    """
    from factor_construction import get_factor_matrix_for_gmm

    results = []
    dates = factor_matrix.index
    step = pd.DateOffset(months=step_months)
    window = pd.DateOffset(years=train_years)

    # Collect all refit dates
    refit_dates = []
    current = dates[0] + window
    while current <= dates[-1]:
        refit_dates.append(current)
        current += step

    if verbose:
        print(f"[rolling_regime] {len(refit_dates)} refits over "
              f"{dates[0].date()} → {dates[-1].date()}")

    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

    for refit_date in refit_dates:
        train_start = refit_date - window
        train_mask  = (dates >= train_start) & (dates < refit_date)
        oos_mask    = (dates >= refit_date) & (dates < refit_date + step)

        X_train = factor_matrix[train_mask].dropna()
        X_oos   = factor_matrix[oos_mask].dropna()

        if len(X_train) < 100 or len(X_oos) == 0:
            continue

        scaler = StandardScaler()
        X_tr_std  = scaler.fit_transform(X_train.values)
        X_oos_std = scaler.transform(X_oos.values)

        gmm = GaussianMixture(
            n_components=n_components, covariance_type="full",
            n_init=n_init, random_state=random_state,
            max_iter=500, reg_covar=1e-6,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gmm.fit(X_tr_std)

        probs = gmm.predict_proba(X_oos_std)
        df_probs = pd.DataFrame(
            probs,
            index=X_oos.index,
            columns=[f"Regime_{k}" for k in range(n_components)],
        )
        results.append(df_probs)

    if not results:
        return pd.DataFrame()

    return pd.concat(results).sort_index()
