import math

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from src.features import cumulative_returns


def plot_cumulative_returns(returns, save_path=None):
    """Plot cumulative returns for all factor proxies."""
    cumulative = cumulative_returns(returns)
    fig, ax = plt.subplots(figsize=(12, 6))
    cumulative.plot(ax=ax)
    ax.set_title("Cumulative Returns by Factor Proxy")
    ax.set_ylabel("Cumulative return")
    ax.set_xlabel("")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    save_figure(fig, save_path)
    return fig, ax


def plot_regime_colored_series(cumulative, labels, column="Equity", save_path=None, model_name=None):
    """Plot one cumulative return series colored by inferred regime."""
    if model_name is None:
        model_name = "HMM" if labels.name == "HiddenState" else "GMM"
    label_name = "State" if model_name == "HMM" else "Regime"

    fig, ax = plt.subplots(figsize=(12, 6))

    for regime in sorted(labels.unique()):
        regime_data = cumulative[column].where(labels == regime)
        ax.scatter(
            regime_data.index,
            regime_data,
            s=8,
            label=f"{label_name} {regime}",
        )

    ax.plot(cumulative.index, cumulative[column], color="black", linewidth=0.8, alpha=0.35)
    ax.set_title(f"{model_name} {column} Cumulative Return Colored by {label_name}")
    ax.set_ylabel("Cumulative return")
    ax.set_xlabel("")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    save_figure(fig, save_path)
    return fig, ax


def plot_regime_probabilities(probabilities, save_path=None, model_name=None):
    """Plot GMM regime probabilities through time."""
    if model_name is None:
        model_name = "HMM" if str(probabilities.columns[0]).startswith("State") else "GMM"
    label_name = "State" if model_name == "HMM" else "Regime"

    fig, ax = plt.subplots(figsize=(12, 6))
    probabilities.plot(ax=ax)
    ax.set_title(f"{model_name} {label_name} Probabilities")
    ax.set_ylabel("Probability")
    ax.set_xlabel("")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, save_path)
    return fig, ax


def plot_regime_correlation_heatmaps(correlations, save_path=None, model_name="GMM"):
    """Plot one average correlation heatmap for each regime."""
    label_name = "State" if model_name == "HMM" else "Regime"
    n_regimes = len(correlations)
    n_cols = 2
    n_rows = math.ceil(n_regimes / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, (regime, corr) in zip(axes, correlations.items()):
        sns.heatmap(
            corr,
            ax=ax,
            vmin=-1,
            vmax=1,
            cmap="vlag",
            center=0,
            annot=True,
            fmt=".2f",
            square=True,
            cbar=True,
        )
        ax.set_title(f"{model_name} {label_name} {regime} Average Correlations")

    for ax in axes[len(correlations) :]:
        ax.axis("off")

    fig.tight_layout()
    save_figure(fig, save_path)
    return fig, axes


def plot_factor_volatility_table(summary, save_path=None, model_name="GMM"):
    """Plot annualized factor volatilities by regime as a presentation-style table."""
    label_name = "State" if model_name == "HMM" else "Regime"
    return plot_factor_metric_table(
        summary,
        metric_suffix="Volatility",
        title=f"{model_name} Factor Volatilities by {label_name}",
        subtitle=(
            "Annualized standard deviation of daily returns. "
            "Low volatility means calmer behavior; high volatility means larger, less stable moves."
        ),
        save_path=save_path,
        color_mode="low_to_high",
        model_name=model_name,
    )


def plot_factor_return_table(summary, save_path=None, model_name="GMM"):
    """Plot annualized factor returns by regime as a presentation-style table."""
    label_name = "State" if model_name == "HMM" else "Regime"
    return plot_factor_metric_table(
        summary,
        metric_suffix="Return",
        title=f"{model_name} Factor Returns by {label_name}",
        subtitle="Annualized average return for each factor inside each market condition.",
        save_path=save_path,
        color_mode="bad_to_good",
        model_name=model_name,
    )


def plot_factor_metric_table(
    summary,
    metric_suffix,
    title,
    subtitle,
    save_path=None,
    color_mode="low_to_high",
    model_name="GMM",
):
    """Plot a grouped factor table for one regime summary metric."""
    label_name = "State" if model_name == "HMM" else "Regime"
    factor_groups = {
        "Core Macro": [
            ("Equity", "Equity"),
            ("Interest_Rate", "Interest Rates"),
            ("Credit", "Credit"),
            ("Commodities", "Commodities"),
        ],
        "Secondary Macro": [
            ("Emerging_Market", "Emerging Markets"),
            ("Foreign_Currency", "Foreign Currency"),
            ("Local_Inflation", "Local Inflation"),
            ("Local_Equity", "Local Equity"),
        ],
    }

    rows = []
    matched_factors = set()
    for group, factors in factor_groups.items():
        for factor, display_name in factors:
            column = f"{factor}_{metric_suffix}"
            if column in summary.columns:
                rows.append((group, display_name, column))
                matched_factors.add(factor)

    suffix = f"_{metric_suffix}"
    for column in summary.columns:
        if not column.endswith(suffix):
            continue
        factor = column[: -len(suffix)]
        if factor in matched_factors:
            continue
        display_name = factor.replace("_", " ")
        rows.append(("Other Factors", display_name, column))

    if not rows:
        raise ValueError(f"No {metric_suffix.lower()} columns were found in the regime summary.")

    regimes = list(summary.index)
    values = np.array([[summary.loc[regime, column] * 100 for regime in regimes] for _, _, column in rows])
    if color_mode == "bad_to_good":
        max_abs = max(abs(np.nanmin(values)), abs(np.nanmax(values)))
        vmin, vmax = -max_abs, max_abs
        cmap = LinearSegmentedColormap.from_list("regime_returns", ["#ff1f4f", "#ffdf57", "#86bf70"])
    else:
        vmin = np.nanmin(values)
        vmax = np.nanmax(values)
        cmap = LinearSegmentedColormap.from_list("regime_metric", ["#86bf70", "#ffdf57", "#ff1f4f"])

    n_rows, n_regimes = values.shape
    group_width = 0.8
    factor_width = 2.05
    regime_width = 1.5
    heat_start = group_width + factor_width
    total_width = heat_start + n_regimes * regime_width
    row_height = 1.0
    header_height = 1.45

    fig_width = max(12, total_width * 1.2)
    fig_height = max(5.2, n_rows * 0.55 + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlim(0, total_width)
    ax.set_ylim(0, n_rows + header_height)
    ax.axis("off")

    ax.text(
        total_width / 2,
        n_rows + header_height + 0.45,
        title,
        ha="center",
        va="bottom",
        fontsize=18,
        fontweight="bold",
    )
    ax.text(
        total_width / 2,
        n_rows + header_height + 0.08,
        subtitle,
        ha="center",
        va="bottom",
        fontsize=10,
        color="#333333",
    )

    for j, regime in enumerate(regimes):
        ax.text(
            heat_start + j * regime_width + regime_width / 2,
            n_rows + 0.35,
            f"{label_name} {regime}",
            ha="center",
            va="center",
            fontsize=11,
        )

    if vmin == vmax:
        vmax = vmin + 1

    for i, (group, factor_name, _) in enumerate(rows):
        y = n_rows - i - 1
        ax.add_patch(Rectangle((group_width, y), factor_width, row_height, facecolor="white", edgecolor="white"))
        ax.text(group_width + 0.05, y + 0.5, factor_name, ha="left", va="center", fontsize=12)

        for j in range(n_regimes):
            color = cmap((values[i, j] - vmin) / (vmax - vmin))
            x = heat_start + j * regime_width
            ax.add_patch(Rectangle((x, y), regime_width, row_height, facecolor=color, edgecolor="none"))
            ax.text(
                x + regime_width / 2,
                y + 0.5,
                f"{values[i, j]:.2f}%",
                ha="center",
                va="center",
                fontsize=12,
                color="black",
            )

    group_start = 0
    while group_start < n_rows:
        group = rows[group_start][0]
        group_end = group_start
        while group_end < n_rows and rows[group_end][0] == group:
            group_end += 1

        y_bottom = n_rows - group_end
        height = (group_end - group_start) * row_height
        ax.add_patch(Rectangle((0, y_bottom), group_width, height, facecolor="black", edgecolor="white", linewidth=1.5))
        ax.text(
            group_width / 2,
            y_bottom + height / 2,
            group,
            ha="center",
            va="center",
            color="white",
            fontsize=11,
            rotation=90,
        )
        group_start = group_end

    ax.add_patch(Rectangle((0, 0), total_width, n_rows, facecolor="none", edgecolor="black", linewidth=1.0))
    ax.plot([0, total_width], [n_rows, n_rows], color="black", linewidth=1.0)
    for x in [group_width, heat_start, total_width]:
        ax.plot([x, x], [0, n_rows], color="black", linewidth=1.0)

    fig.tight_layout()
    save_figure(fig, save_path)
    return fig, ax




def plot_model_selection(selection_table, save_path=None):
    """Plot GMM AIC and BIC across component counts."""
    fig, ax = plt.subplots(figsize=(9, 5))
    selection_table[["AIC", "BIC"]].plot(marker="o", ax=ax)
    ax.set_title("GMM Model Selection")
    ax.set_xlabel("Number of regimes")
    ax.set_ylabel("Information criterion")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, save_path)
    return fig, ax


def plot_transition_heatmap(matrix, title="Regime Transition Matrix", save_path=None, fmt=".2f", label_name=None):
    """Plot a transition count or probability matrix."""
    if label_name is None:
        label_name = "state" if "HMM" in title else "regime"

    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(matrix, annot=True, fmt=fmt, cmap="Blues", cbar=True, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(f"To {label_name}")
    ax.set_ylabel(f"From {label_name}")
    fig.tight_layout()
    save_figure(fig, save_path)
    return fig, ax


def plot_regime_timeline_comparison(gmm_labels, hmm_labels, save_path=None):
    """Plot GMM and HMM labels on aligned timelines."""
    if not gmm_labels.index.equals(hmm_labels.index):
        raise ValueError("GMM and HMM labels must have matching indices.")

    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    axes[0].scatter(gmm_labels.index, gmm_labels, c=gmm_labels, cmap="tab10", s=8)
    axes[0].set_title("GMM Regimes")
    axes[0].set_ylabel("Regime")
    axes[0].grid(True, alpha=0.25)

    axes[1].scatter(hmm_labels.index, hmm_labels, c=hmm_labels, cmap="tab10", s=8)
    axes[1].set_title("HMM Hidden States")
    axes[1].set_ylabel("State")
    axes[1].set_xlabel("")
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    save_figure(fig, save_path)
    return fig, axes


def save_figure(fig, save_path):
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
