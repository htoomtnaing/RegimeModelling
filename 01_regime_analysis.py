# %% [markdown]
# # Regime Modelling — Analysis Notebook
# 
# Two Sigma GMM approach · Bloomberg + Fama-French data
# 
# **Run cells top-to-bottom in order. Each section saves its outputs to `outputs/GMM/`.**

# %% [markdown]
# ## 0. Setup

# %%
import sys, os, warnings, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')   # safe for all environments; switch to 'inline' in Jupyter
# %matplotlib inline

pd.set_option('display.float_format', '{:.4f}'.format)
pd.set_option('display.max_columns', 30)
plt.rcParams.update({'figure.dpi': 120, 'font.size': 10})

OUT_ROOT = ROOT / 'outputs' / 'GMM'
FIG_ROOT = OUT_ROOT / 'figures'
OUT_ROOT.mkdir(parents=True, exist_ok=True)
FIG_ROOT.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1. Configure data paths
# 
# Set `DATA_ROOT` to the folder containing `Bloomberg_Data/`, `Farma_French/`, `Macro/`.

# %%
DATA_ROOT = ROOT / 'Data'

from src import data_loader as dl
dl.DATA_ROOT     = DATA_ROOT
dl.BLOOMBERG_DIR = DATA_ROOT / 'Bloomberg_Data'
dl.MACRO_DIR     = DATA_ROOT / 'Macro'
dl.FF_DIR        = DATA_ROOT / 'Farma_French'

for subdir in ['Bloomberg_Data', 'Farma_French', 'Macro']:
    p = DATA_ROOT / subdir
    status = 'OK' if p.exists() else 'MISSING — update DATA_ROOT'
    print(f'  {subdir:20s}  {status}')

# %% [markdown]
# ## 2. Load & inspect raw data

# %%
from src.data_loader import (
    load_bloomberg_all, load_ff5_daily,
    load_interest_rate_daily, load_cpi_interest_rates,
    ALL_BLOOMBERG_TICKERS,
)

raw_panel = load_bloomberg_all(ALL_BLOOMBERG_TICKERS)
ff5 = load_ff5_daily()
ir  = load_interest_rate_daily()
cpi = load_cpi_interest_rates()

print(f'Bloomberg panel: {raw_panel.shape}')
print(f'FF5 daily:       {ff5.shape}  {ff5.index[0].date()} -> {ff5.index[-1].date()}')
print(f'Interest rates:  {ir.shape}')
print(f'CPI quarterly:   {cpi.shape}')
print()
print('Bloomberg series coverage:')
for col in raw_panel.columns:
    s = raw_panel[col].dropna()
    print(f'  {col:12s}  {len(s):5d} obs  {s.index[0].date()} -> {s.index[-1].date()}')

# %% [markdown]
# ## 3. Build clean data panel
# 
# Align to business-day calendar, forward-fill gaps, compute log returns.
# 
# | Start date | Coverage | Notes |
# |-----------|----------|-------|
# | `2002-01-02` | All 15 factors daily | **Recommended** |
# | `1999-01-04` | Slightly longer, swap MXCXDMHR→MXWD | |
# | `1987-07-01` | Longest history, monthly pre-2001 | |

# %%
from src.data_cleaner import build_clean_dataset

START_DATE = '2002-01-02'

t0 = time.time()
dataset  = build_clean_dataset(start=START_DATE)
prices   = dataset['prices']
returns  = dataset['returns']
ff5_aln  = dataset['ff5']
rf       = dataset['rf']
calendar = dataset['calendar']
print(f'Built in {time.time()-t0:.1f}s')
print(f'Calendar : {calendar[0].date()} -> {calendar[-1].date()}  ({len(calendar)} days)')
print(f'Returns  : {returns.shape}')
print(f'RF (mean daily): {rf.mean()*100:.5f}%')

ff5.to_csv(OUT_ROOT / 'ff5_daily.csv')
rf.to_csv(OUT_ROOT / 'risk_free_daily.csv')
print(f'Saved: {OUT_ROOT / "ff5_daily.csv"}')
print(f'Saved: {OUT_ROOT / "risk_free_daily.csv"}')

# Save
prices.to_parquet(OUT_ROOT / 'prices.parquet')
returns.to_parquet(OUT_ROOT / 'returns.parquet')
print(f'Saved: {OUT_ROOT / "prices.parquet"}, {OUT_ROOT / "returns.parquet"}')

# %%
# Sanity check: cumulative returns of core series
fig, axes = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
checks = [
    ('MXCXDMHR', 'Equity (MSCI ACWI Hedged)'),
    ('LGY7TRUH', 'Interest Rates (Global Govt 7-10yr)'),
    ('LUACTRUU', 'Credit IG (US Corporate)'),
    ('BCOMTR',   'Commodities (BCOM TR)'),
]
for ax, (col, label) in zip(axes, checks):
    if col in returns.columns:
        cum = (1 + returns[col]).cumprod()
        ax.plot(cum.index, cum.values, linewidth=0.8)
        ax.set_ylabel(label, fontsize=8)
        ax.grid(alpha=0.3)
axes[-1].set_xlabel('Date')
fig.suptitle('Cumulative Returns — Raw Series', fontsize=12)
plt.tight_layout()
fig.savefig(FIG_ROOT / '01_raw_cumulative_returns.png', dpi=120, bbox_inches='tight')
plt.show()
print(f'Saved: {FIG_ROOT / "01_raw_cumulative_returns.png"}')

# %% [markdown]
# ## 4. Construct factors
# 
# Core Macro → Secondary Macro → Style (FF5). Each later group is residualised against all earlier factors (rolling EWM-OLS, 60-day half-life).

# %%
from src.factor_construction import build_factor_matrix, get_factor_matrix_for_gmm

t0 = time.time()
factor_matrix = build_factor_matrix(
    start=START_DATE,
    include_style=True,
    halflife_days=60,
    min_periods=126,
    verbose=True,
)
print(f'\nFactor construction: {time.time()-t0:.1f}s')

# Save
factor_matrix.to_parquet(OUT_ROOT / 'factor_matrix.parquet')
print(f'Saved: {OUT_ROOT / "factor_matrix.parquet"}')

# %%
# Summary statistics
print(f'Factor matrix: {factor_matrix.shape}')
print(f'Date range   : {factor_matrix.index[0].date()} -> {factor_matrix.index[-1].date()}')
print()
ann_mean = factor_matrix.mean() * 252 * 100
ann_vol  = factor_matrix.std()  * (252**0.5) * 100
sharpe   = ann_mean / ann_vol
summary  = pd.DataFrame({'Mean_%': ann_mean, 'Vol_%': ann_vol, 'Sharpe': sharpe})
print(summary.round(3).to_string())
summary.to_csv(OUT_ROOT / 'factor_summary_stats.csv')
print(f'\nSaved: {OUT_ROOT / "factor_summary_stats.csv"}')

# %%
# Factor correlation matrix (orthogonality check — should be near-zero off-diagonal)
corr = factor_matrix.dropna().corr()
fig, ax = plt.subplots(figsize=(13, 11))
im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
ax.set_xticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=8)
ax.set_yticks(range(len(corr.index)))
ax.set_yticklabels(corr.index, fontsize=8)
ax.set_title('Factor Correlation Matrix (target: near-zero off-diagonal)', fontsize=11)
for i in range(len(corr)):
    for j in range(len(corr.columns)):
        ax.text(j, i, f'{corr.values[i,j]:.2f}', ha='center', va='center', fontsize=6)
plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
fig.savefig(FIG_ROOT / '02_factor_correlation.png', dpi=120, bbox_inches='tight')
plt.show()
corr.to_csv(OUT_ROOT / 'factor_correlation.csv')
print(f'Saved: {FIG_ROOT / "02_factor_correlation.png"}, {OUT_ROOT / "factor_correlation.csv"}')

# %% [markdown]
# ## 5. Prepare GMM input matrix
# 
# `get_factor_matrix_for_gmm` drops any remaining NaN rows (from the burn-in period). This is the matrix fed directly into the GMM — keep it as `X` for all subsequent steps.

# %%
X = get_factor_matrix_for_gmm(factor_matrix, dropna=True)
print(f'GMM input X: {X.shape}')
print(f'Date range : {X.index[0].date()} -> {X.index[-1].date()}')
print(f'Dropped    : {len(factor_matrix) - len(X)} rows (NaN burn-in)')

X.to_parquet(OUT_ROOT / 'gmm_input_X.parquet')
print(f'Saved: {OUT_ROOT / "gmm_input_X.parquet"}')

# %% [markdown]
# ## 6. Fit the GMM regime model
# 
# **Option A** (`run_cv=False`, `n_components=4`): directly reproduces the Two Sigma four-regime result. Runs in ~30 seconds.
# 
# **Option B** (`run_cv=True`): cross-validates over n=2..6 and uses elbow detection to pick the optimal number. Adds ~2–3 minutes. Use the CV plot to sanity-check the selection.

# %%
from src.regime_model import fit_regime_model, predict_current_regime

t0 = time.time()
model = fit_regime_model(
    X,
    n_components=4,        # ← change to None and run_cv=True to auto-select
    n_components_range=range(2, 7),
    n_init=20,
    cv_splits=5,
    run_cv=False,          # ← set True to run cross-validation (~2-3 min extra)
    verbose=True,
)
print(f'\nGMM fitting: {time.time()-t0:.1f}s')

# Save probabilities and labels immediately
model.probabilities.to_parquet(OUT_ROOT / 'regime_probabilities.parquet')
model.hard_labels.rename('regime_int').to_frame().to_parquet(OUT_ROOT / 'regime_hard_labels.parquet')
print(f'Saved: {OUT_ROOT / "regime_probabilities.parquet"}')
print(f'Saved: {OUT_ROOT / "regime_hard_labels.parquet"}')

# %%
# CV score plot (only populated when run_cv=True)
from src.regime_utils import plot_cv_scores
if model.cv_scores:
    fig = plot_cv_scores(model.cv_scores)
    fig.savefig(FIG_ROOT / '03_cv_scores.png', dpi=120, bbox_inches='tight')
    plt.show()
    print(f'Saved: {FIG_ROOT / "03_cv_scores.png"}')
else:
    print('CV not run (run_cv=False). Set run_cv=True above to generate this plot.')
print(f'n_components = {model.n_components}')

# %% [markdown]
# ## 7. Label regimes
# 
# The auto-labeller applies Two Sigma heuristics (most-negative equity = Crisis, highest Local Inflation = Inflation, highest equity vol among positive-return regimes = WOI). **Always verify against the heatmap below and override if needed.**

# %%
from src.regime_utils import compute_regime_stats, plot_factor_heatmap, relabel_regimes

# compute_regime_stats aligns factor_matrix and hard_labels on common index automatically
means_df, vols_df = compute_regime_stats(factor_matrix, model.hard_labels, model.regime_names)

print('Auto-assigned labels:', model.regime_names)
print()
print('Annualised factor means by regime (%):')
print((means_df * 100).round(2).to_string())

means_df.to_csv(OUT_ROOT / 'regime_factor_means.csv')
vols_df.to_csv(OUT_ROOT / 'regime_factor_vols.csv')
print(f'\nSaved: {OUT_ROOT / "regime_factor_means.csv"}, {OUT_ROOT / "regime_factor_vols.csv"}')

# %%
fig = plot_factor_heatmap(
    means_df, vols_df,
    title='Annualised Factor Mean Returns by Regime (%)'
)
fig.savefig(FIG_ROOT / '04_factor_heatmap.png', dpi=120, bbox_inches='tight')
plt.show()
print(f'Saved: {FIG_ROOT / "04_factor_heatmap.png"}')

# %%
# ── MANUAL RELABELLING ────────────────────────────────────────────────
# After inspecting the heatmap, override labels if the auto-labeller got any wrong.
# Map each cluster INDEX (integer) to a name string.
#
# Example for 4 components — adjust indices based on your heatmap:
# model = relabel_regimes(model, {
#     0: 'Crisis',
#     1: 'Steady_State',
#     2: 'Inflation',
#     3: 'WOI',
# })
#
# Recompute stats after relabelling:
# means_df, vols_df = compute_regime_stats(factor_matrix, model.hard_labels, model.regime_names)

print('Current labels:', model.regime_names)

# Save named labels
named_labels = model.hard_labels.map(model.regime_names)
named_labels.name = 'regime'
named_labels.to_frame().to_parquet(OUT_ROOT / 'regime_named_labels.parquet')
named_labels.to_frame().to_csv(OUT_ROOT / 'regime_named_labels.csv')
print(f'Saved: {OUT_ROOT / "regime_named_labels.parquet"}, {OUT_ROOT / "regime_named_labels.csv"}')

# %% [markdown]
# ## 8. Historical regime analysis

# %%
from src.regime_utils import plot_regime_timeline, plot_regime_probabilities, get_regime_periods

fig = plot_regime_timeline(
    model.hard_labels, model.regime_names,
    title='Regime Classification History', figsize=(16, 2.5)
)
fig.savefig(FIG_ROOT / '05_regime_timeline.png', dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved: {FIG_ROOT / "05_regime_timeline.png"}')

# %%
fig = plot_regime_probabilities(
    model.probabilities, model.regime_names,
    title='Regime Probabilities Over Time', figsize=(16, 4)
)
fig.savefig(FIG_ROOT / '06_regime_probabilities.png', dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved: {FIG_ROOT / "06_regime_probabilities.png"}')

# %%
freq = model.hard_labels.value_counts().sort_index()
print('Regime frequency:')
rows = []
for k, cnt in freq.items():
    pct   = 100 * cnt / len(model.hard_labels)
    label = model.regime_names.get(k, f'Regime_{k}')
    print(f'  {label:20s}  {cnt:5d} days  ({pct:.1f}%)')
    rows.append({'regime': label, 'days': cnt, 'pct': round(pct, 2)})
freq_df = pd.DataFrame(rows)
freq_df.to_csv(OUT_ROOT / 'regime_frequency.csv', index=False)
print(f'\nSaved: {OUT_ROOT / "regime_frequency.csv"}')

periods = get_regime_periods(model.hard_labels, model.regime_names)
print(f'{len(periods)} contiguous regime periods (first 20):')
print(periods.head(20).to_string(index=False))

periods.to_csv(OUT_ROOT / 'regime_periods.csv', index=False)
print(f'\nSaved: {OUT_ROOT / "regime_periods.csv"}')

# %% [markdown]
# ## 9. Transition matrix

# %%
from src.regime_utils import compute_transition_matrix

trans = compute_transition_matrix(model.hard_labels, model.regime_names)
print('Regime transition probabilities (%):')
print((trans * 100).round(1).to_string())

trans.to_csv(OUT_ROOT / 'regime_transition_matrix.csv')
print(f'\nSaved: {OUT_ROOT / "regime_transition_matrix.csv"}')

# %% [markdown]
# ## 10. Current regime — "Where are we now?"

# %%
# Uses factor_matrix (with NaN) — predict_current_regime calls dropna() internally
current = predict_current_regime(model, factor_matrix, window_days=60)
print('Current regime probabilities (last 60 days):')
print(current.to_string())

recent = factor_matrix.tail(252).mean() * 252 * 100
print()
print('Factor returns (ann. %, trailing 1yr):')
print(recent.sort_values(ascending=False).round(2).to_string())

current.to_frame('probability').to_csv(OUT_ROOT / 'current_regime.csv')
print(f'\nSaved: {OUT_ROOT / "current_regime.csv"}')

# %% [markdown]
# ## 11. Full dashboard

# %%
from src.regime_utils import plot_dashboard
fig = plot_dashboard(model, factor_matrix, figsize=(18, 14))
fig.savefig(FIG_ROOT / '07_dashboard.png', dpi=120, bbox_inches='tight')
plt.show()
print(f'Saved: {FIG_ROOT / "07_dashboard.png"}')

# %% [markdown]
# ## 12. Walk-forward stability (optional — slow)
# 
# Re-fits the GMM on rolling 5-year windows to check regime-label consistency. Uncomment to run — expect 10–20 minutes.

# %%
# from regime_utils import rolling_regime_window
#
# t0 = time.time()
# rolling_probs = rolling_regime_window(
#     X, n_components=model.n_components,
#     train_years=5, step_months=3, n_init=10, verbose=True,
# )
# print(f'Walk-forward: {time.time()-t0:.0f}s')
# rolling_probs.to_parquet(OUT_ROOT / 'rolling_regime_probs.parquet')
# print(f'Saved: {OUT_ROOT / "rolling_regime_probs.parquet"}')
print('Walk-forward disabled. Uncomment the block above to run.')

# %% [markdown]
# ## 13. Output summary
# 
# All files saved to `outputs/GMM/`:
# 
# | File | Contents |
# |------|----------|
# | `prices.parquet` | Aligned daily price panel |
# | `returns.parquet` | Daily log-return panel |
# | `factor_matrix.parquet` | Full 15-factor return matrix (with NaN burn-in) |
# | `gmm_input_X.parquet` | Clean factor matrix fed into GMM (no NaN) |
# | `regime_probabilities.parquet` | P(regime_k \| day_t) for all k, t |
# | `regime_hard_labels.parquet` | Integer cluster label per day |
# | `regime_named_labels.parquet/.csv` | Named label per day (e.g. 'Crisis') |
# | `regime_frequency.csv` | Days and percentages per regime |
# | `regime_factor_means.csv` | Annualised mean return per factor per regime |
# | `regime_factor_vols.csv` | Annualised vol per factor per regime |
# | `regime_periods.csv` | Start/end dates of each contiguous regime period |
# | `regime_transition_matrix.csv` | Transition probability matrix |
# | `current_regime.csv` | Latest regime probabilities |
# | `factor_summary_stats.csv` | Mean/vol/Sharpe per factor |
# | `factor_correlation.csv` | Factor correlation matrix |
# | `figures/` | All plots as PNG |
# 
# **Next step:** load `regime_probabilities.parquet` into the trading module to build regime-conditional signals.

# %%
# Verify all outputs were created
import os
print(f'{OUT_ROOT.relative_to(ROOT)} contents:')
for f in sorted(os.listdir(OUT_ROOT)):
    full = os.path.join(OUT_ROOT, f)
    if os.path.isfile(full):
        size_kb = os.path.getsize(full) / 1024
        print(f'  {f:45s}  {size_kb:7.1f} KB')
print()
print(f'{FIG_ROOT.relative_to(ROOT)} contents:')
for f in sorted(os.listdir(FIG_ROOT)):
    full = os.path.join(FIG_ROOT, f)
    if os.path.isfile(full):
        size_kb = os.path.getsize(full) / 1024
        print(f'  {f:45s}  {size_kb:7.1f} KB')


