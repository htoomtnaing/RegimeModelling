"""
portfolio_optimiser.py  —  Regime-conditional CVaR / DRO  (long-short)
=======================================================================

Long-Short CVaR/DRO Formulation
---------------------------------
The optimisation problem is:

    min   CVaR_alpha(w)
    s.t.  sum(w)  = 1          (net fully invested)

No per-asset weight bounds. No gross leverage cap.
The only constraint is that weights sum to 1.

This is implemented as a LINEAR PROGRAMME using the Rockafellar-Uryasev
(2000) reformulation:

    min   VaR + 1/((1-alpha)*S) * sum_s(z_s)
    s.t.  z_s >= -r_s'w - VaR    for each scenario s
          z_s >= 0
          sum(w) = 1
          w unrestricted (free variables — allows shorts)

Solved with scipy linprog (HiGHS backend). This is a proper LP with a
finite, well-posed solution because the tail scenarios anchor the
optimum even with free weights.

DRO adds a Wasserstein penalty eps * ||w||_1 to the LP objective,
which penalises gross exposure and naturally controls leverage without
imposing a hard cap.

Why LP not SLSQP?
-----------------
SLSQP on CVaR without weight bounds is an unbounded problem unless
regularisation (lam) is added, and lam suppresses short positions.
The LP formulation has no such issue: it is always bounded because
the worst-tail-scenario constraint anchors the solution, and it
produces genuine long-short positions determined purely by tail risk.

Regime conditioning
-------------------
Scenario weights are boosted (regime_boost x) for observations matching
the current GMM regime, concentrating the tail on the current environment.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

_VERSION = 'portfolio_optimiser v5 — LP long-only CVaR/DRO'
print(f'[portfolio_optimiser] loaded: {_VERSION}')

# ── Pandas version compatibility ──────────────────────────────────────────────
def _compat_freq(freq: str) -> str:
    major, minor = (int(x) for x in pd.__version__.split('.')[:2])
    if (major, minor) >= (2, 2):
        return freq
    return {'ME':'M','QE':'Q','YE':'A','BME':'BM','BQE':'BQ','BYE':'BA'}.get(freq, freq)

# ── Asset universes ───────────────────────────────────────────────────────────
ASSET_TICKERS = [
    'MXCXDMHR','RU30INTR','MXEF',    'BCOMTR',
    'LGY7TRUH', 'LUACTRUU','LF98TRUU','BCIT5T',
    'EMUSTRUU', 'PUT',
]
ASSET_LABELS = {
    'MXCXDMHR':'Global Equity', 'RU30INTR':'US Equity',
    'MXEF':     'EM Equity',    'BCOMTR':  'Commodities',
    'LGY7TRUH': 'Rates 7-10yr', 'LUACTRUU':'US IG Credit',
    'LF98TRUU': 'US HY Credit', 'BCIT5T':  'US TIPS',
    'EMUSTRUU': 'EM Bonds',     'PUT':     'Short Vol',
}

# ── Result containers ─────────────────────────────────────────────────────────
@dataclass
class OptimResult:
    weights:     np.ndarray
    cvar:        float
    var:         float
    status:      str
    asset_names: list[str] = field(default_factory=list)

    def to_series(self) -> pd.Series:
        return pd.Series(self.weights, index=self.asset_names)

@dataclass
class BacktestResult:
    portfolio_returns: pd.DataFrame
    weights_history:   dict
    metrics:           pd.DataFrame
    regime_labels:     pd.Series


# ── CVaR LP (pure, no weight constraints) ─────────────────────────────────────
def optimise_cvar(
    scenarios: np.ndarray,          # (S, N) scenario return matrix
    alpha:     float = 0.95,        # CVaR confidence level
) -> OptimResult:
    """
    Minimise CVaR via Rockafellar-Uryasev LP.

    Variables: [w_1..w_N,  VaR,  z_1..z_S]
    w >= 0 (long-only), no upper bound per asset.

    Constraints:
        sum(w) = 1              (net fully invested)
        w_i >= 0                (long-only)
        z_s >= -r_s'w - VaR    (tail loss slacks, one per scenario)
        z_s >= 0
    """
    # Guard: remove any NaN/Inf rows from scenario matrix
    finite_mask = np.isfinite(scenarios).all(axis=1)
    if not finite_mask.all():
        scenarios = scenarios[finite_mask]
    if len(scenarios) == 0:
        return OptimResult(weights=np.ones(scenarios.shape[1])/scenarios.shape[1],
                           cvar=0.0, var=0.0, status='failed:all_nan')

    S, N    = scenarios.shape
    n_vars  = N + 1 + S

    # Objective: min VaR + 1/((1-alpha)*S) * sum(z)
    c       = np.zeros(n_vars)
    c[N]    = 1.0
    c[N+1:] = 1.0 / ((1 - alpha) * S)

    # Inequality: -r_s'w - VaR - z_s <= 0
    A_ub        = np.zeros((S, n_vars))
    A_ub[:, :N] = -scenarios
    A_ub[:, N]  = -1.0
    for s in range(S):
        A_ub[s, N + 1 + s] = -1.0
    b_ub = np.zeros(S)

    # Equality: sum(w) = 1
    A_eq       = np.zeros((1, n_vars))
    A_eq[0,:N] = 1.0
    b_eq       = np.array([1.0])

    # Bounds: w >= 0 (long-only), VaR FREE, z >= 0
    bounds = [(0, None)] * N + [(None, None)] + [(0, None)] * S

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                      A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method='highs')

    if res.status != 0:
        # Fallback: equal weight
        w_opt = np.ones(N) / N
        status = f'failed({res.message}):equal_weight'
    else:
        w_opt  = res.x[:N]
        status = 'optimal'

    losses   = -scenarios @ w_opt
    var_val  = np.percentile(losses, alpha * 100)
    tail     = losses[losses >= var_val]
    cvar_val = float(np.mean(tail)) if len(tail) > 0 else var_val

    return OptimResult(
        weights = w_opt,
        cvar    = cvar_val * np.sqrt(252),
        var     = var_val  * np.sqrt(252),
        status  = status,
    )


# ── DRO (Wasserstein) ─────────────────────────────────────────────────────────
def optimise_dro(
    scenarios: np.ndarray,
    alpha:     float = 0.95,
    kappa:     float = 1.0,
) -> OptimResult:
    """
    Wasserstein DRO with long-only weights.

    Under long-only constraints sum(|w|) = sum(w) = 1, so the
    Wasserstein penalty eps * ||w||_1 = eps (a constant).
    The optimal weights are therefore identical to CVaR; the DRO
    bound is CVaR + eps * sqrt(252) (annualised).

    eps = kappa * sigma_hat / sqrt(S)
    """
    # Guard: remove any NaN/Inf rows from scenario matrix
    finite_mask = np.isfinite(scenarios).all(axis=1)
    if not finite_mask.all():
        scenarios = scenarios[finite_mask]
    if len(scenarios) == 0:
        return OptimResult(weights=np.ones(scenarios.shape[1])/scenarios.shape[1],
                           cvar=0.0, var=0.0, status='failed:all_nan:dro')

    S, N   = scenarios.shape
    sigma  = float(np.std(scenarios))
    eps    = kappa * sigma / np.sqrt(S)

    # Weights identical to CVaR (long-only), DRO bound shifted by eps
    base = optimise_cvar(scenarios, alpha=alpha)

    losses   = -scenarios @ base.weights
    var_val  = np.percentile(losses, alpha * 100)
    tail     = losses[losses >= var_val]
    cvar_val = float(np.mean(tail)) if len(tail) > 0 else var_val

    return OptimResult(
        weights = base.weights,
        cvar    = cvar_val * np.sqrt(252) + eps * np.sqrt(252),
        var     = var_val  * np.sqrt(252),
        status  = base.status + ':dro',
    )


# ── Scenario construction ─────────────────────────────────────────────────────
def build_regime_scenarios(
    returns_history: pd.DataFrame,
    labels_history:  pd.Series,
    current_regime:  int,
    n_scenarios:     int   = 500,
    lookback_days:   int   = 756,
    regime_boost:    float = 3.0,
    random_state:    int   = 42,
) -> np.ndarray:
    """Bootstrap scenarios up-weighted toward the current regime."""
    rng    = np.random.default_rng(random_state)
    common = returns_history.index.intersection(labels_history.index)
    ret    = returns_history.loc[common].tail(lookback_days)
    lbl    = labels_history.loc[common].tail(lookback_days)

    # Drop any rows with NaN/Inf — these cause linprog to fail
    valid  = np.isfinite(ret.values).all(axis=1)
    ret    = ret.iloc[valid]
    lbl    = lbl.iloc[valid]

    if len(ret) == 0:
        raise ValueError("No valid (non-NaN) rows in scenario window.")

    wts    = np.where(lbl.values == current_regime, regime_boost, 1.0)
    wts    = wts / wts.sum()
    idx    = rng.choice(len(ret), size=n_scenarios, replace=True, p=wts)
    return ret.values[idx]


# ── Benchmark strategies (long-only) ─────────────────────────────────────────
def weights_equal_weight(n: int) -> np.ndarray:
    return np.ones(n) / n

def weights_risk_parity(returns_window: pd.DataFrame) -> np.ndarray:
    vols    = returns_window.std().replace(0, np.nan).fillna(returns_window.std().mean())
    inv_vol = 1.0 / vols
    return (inv_vol / inv_vol.sum()).values

def weights_mvo(returns_window: pd.DataFrame) -> np.ndarray:
    N   = len(returns_window.columns)
    mu  = returns_window.mean().values * 252
    cov = returns_window.cov().values  * 252
    def neg_sharpe(w):
        return -(w @ mu) / (np.sqrt(w @ cov @ w) + 1e-8)
    res = minimize(neg_sharpe, np.ones(N)/N, method='SLSQP',
                   bounds=[(0, 1)]*N,
                   constraints=[{'type':'eq','fun':lambda w: w.sum()-1}],
                   options={'maxiter':500,'ftol':1e-9})
    w = np.clip(res.x if res.success else np.ones(N)/N, 0, 1)
    return w / w.sum()

def weights_unconditional_cvar(
    returns_window: pd.DataFrame,
    alpha:          float = 0.95,
) -> np.ndarray:
    """CVaR LP on full history (no regime conditioning)."""
    vals = returns_window.dropna().values
    return optimise_cvar(vals, alpha=alpha).weights


# ── Expanding-window GMM refit ────────────────────────────────────────────────
def _refit_regime_model(factor_matrix_full, current_date, n_components=4, n_init=20):
    from regime_model import fit_regime_model
    from factor_construction import get_factor_matrix_for_gmm
    X = get_factor_matrix_for_gmm(
        factor_matrix_full[factor_matrix_full.index <= current_date], dropna=True
    )
    return fit_regime_model(X, n_components=n_components,
                            n_init=n_init, run_cv=False, verbose=False)


# ── Main backtest engine ──────────────────────────────────────────────────────
def run_backtest(
    returns_test:       pd.DataFrame,
    returns_train:      pd.DataFrame,
    hard_labels_train:  pd.Series,
    hard_labels_test:   pd.Series,
    regime_probs_test:  pd.DataFrame,
    regime_names:       dict,
    alpha:              float = 0.95,
    dro_kappa:          float = 1.0,
    rebalance_freq:     str   = 'D',
    lookback_days:      int   = 756,
    n_scenarios:        int   = 500,
    regime_boost:       float = 5.0,
    random_state:       int   = 42,
    expanding_window:   bool  = False,
    factor_matrix_full: Optional[pd.DataFrame] = None,
    n_components:       int   = 4,
    verbose:            bool  = True,
) -> BacktestResult:
    """
    Walk-forward backtest. CVaR and DRO use pure LP long-short with NO
    per-asset weight bounds and NO gross leverage cap. Benchmarks are
    long-only for comparison.
    """
    assets = list(returns_test.columns)
    N      = len(assets)
    daily  = rebalance_freq == 'D'

    rebal_dates = (
        set(returns_test.index) if daily
        else set(returns_test.resample(_compat_freq(rebalance_freq)).last().index)
    )

    strategies = [
        'CVaR (Regime)', 'DRO (Regime)', 'CVaR (Unconditional)',
        'Equal Weight',  'Risk Parity',  'MVO',
    ]
    port_rets   = {s: [] for s in strategies}
    wts_history = {s: [] for s in strategies}
    date_index  = []
    current_weights = {s: np.ones(N) / N for s in strategies}

    _last_refit_month = None
    _cached_lbl_train = hard_labels_train.copy()

    for date in returns_test.index:
        r = returns_test.loc[date].values
        for s in strategies:
            port_rets[s].append(current_weights[s] @ r)
        date_index.append(date)

        if date not in rebal_dates:
            continue

        # Current regime
        if date in regime_probs_test.index:
            current_regime = int(regime_probs_test.loc[date].idxmax().split('_')[1])
        else:
            current_regime = 0

        # Expanding-window refit
        if expanding_window and factor_matrix_full is not None:
            mk = (date.year, date.month)
            if mk != _last_refit_month:
                _last_refit_month = mk
                try:
                    m = _refit_regime_model(factor_matrix_full, date, n_components)
                    _cached_lbl_train = m.hard_labels
                except Exception as e:
                    if verbose:
                        print(f'  [expanding] refit failed {date.date()}: {e}')

        # Scenario window
        hist_ret = pd.concat([
            returns_train,
            returns_test.loc[returns_test.index < date],
        ]).tail(lookback_days)[assets]

        hist_lbl = pd.concat([
            _cached_lbl_train,
            hard_labels_test.loc[hard_labels_test.index < date],
        ]).tail(lookback_days)

        # CVaR (regime-conditional, pure LP, no weight constraints)
        scen = build_regime_scenarios(
            hist_ret, hist_lbl, current_regime,
            n_scenarios=n_scenarios, lookback_days=lookback_days,
            regime_boost=regime_boost, random_state=random_state,
        )
        res_cvar = optimise_cvar(scen, alpha=alpha)
        res_cvar.asset_names = assets
        current_weights['CVaR (Regime)'] = res_cvar.weights

        # DRO (regime-conditional, LP + Wasserstein penalty, no weight constraints)
        res_dro = optimise_dro(scen, alpha=alpha, kappa=dro_kappa)
        res_dro.asset_names = assets
        current_weights['DRO (Regime)'] = res_dro.weights

        # CVaR unconditional (pure LP, full history)
        current_weights['CVaR (Unconditional)'] = weights_unconditional_cvar(
            hist_ret, alpha=alpha
        )

        # Benchmarks (long-only)
        current_weights['Equal Weight'] = weights_equal_weight(N)
        current_weights['Risk Parity']  = weights_risk_parity(hist_ret.tail(252))
        current_weights['MVO']          = weights_mvo(hist_ret.tail(252))

        for s in strategies:
            wts_history[s].append(
                pd.Series(current_weights[s], index=assets, name=date)
            )

        if verbose and not daily:
            rn    = regime_names.get(current_regime, f'R{current_regime}')
            w     = pd.Series(res_cvar.weights, index=assets)
            top3  = w.nlargest(3)
            top3_str = ', '.join(
                f'{ASSET_LABELS.get(a,a)}={v:.0%}' for a,v in top3.items()
            )
            print(f'  {date.date()}  [{rn:15s}]  top3: {top3_str}')
        elif verbose and daily and date.day == 1:
            rn   = regime_names.get(current_regime, f'R{current_regime}')
            w    = pd.Series(res_cvar.weights, index=assets)
            top1 = w.idxmax()
            print(f'  {date.date()}  [{rn:15s}]  top: {ASSET_LABELS.get(top1,top1)}={w[top1]:.0%}')

    # Assemble
    port_ret_df = pd.DataFrame(port_rets, index=date_index)
    wts_hist_df = {s: pd.DataFrame(wts_history[s]) for s in strategies if wts_history[s]}
    regime_labels = hard_labels_test.map(regime_names)
    metrics = compute_metrics(port_ret_df, alpha=alpha, regime_labels=regime_labels)

    return BacktestResult(
        portfolio_returns = port_ret_df,
        weights_history   = wts_hist_df,
        metrics           = metrics,
        regime_labels     = regime_labels,
    )


# ── Performance metrics ───────────────────────────────────────────────────────
def compute_metrics(
    port_returns:  pd.DataFrame,
    alpha:         float = 0.95,
    rf_daily:      float = 0.0,
    regime_labels: Optional[pd.Series] = None,
) -> pd.DataFrame:
    rows = []
    td   = 252
    _REGIME_ORDER = ['Crisis', 'Steady_State', 'Inflation', 'WOI']

    for col in port_returns.columns:
        r        = port_returns[col].dropna()
        ann_ret  = r.mean() * td
        ann_vol  = r.std()  * np.sqrt(td)
        sharpe   = ann_ret / (ann_vol + 1e-8)
        down     = (r - rf_daily)[r < rf_daily]
        d_std    = np.sqrt((down**2).mean()) * np.sqrt(td) if len(down) > 0 else 1e-8
        sortino  = ann_ret / (d_std + 1e-8)
        s_r      = np.sort(r.values)
        var_idx  = max(int(np.floor((1 - alpha) * len(s_r))), 1)
        cvar_v   = -s_r[:var_idx].mean() * np.sqrt(td)
        cum      = (1 + r).cumprod()
        dd       = (cum - cum.cummax()) / cum.cummax()
        max_dd   = dd.min()
        calmar   = ann_ret / (-max_dd + 1e-8)

        row = {
            'Ann. Return (%)':             round(ann_ret * 100, 2),
            'Ann. Vol (%)':                round(ann_vol * 100, 2),
            'Sharpe Ratio':                round(sharpe,        3),
            'Sortino Ratio':               round(sortino,       3),
            f'CVaR {int(alpha*100)}% (%)': round(cvar_v  * 100, 2),
            'Max Drawdown (%)':            round(max_dd  * 100, 2),
            'Calmar Ratio':                round(calmar,        3),
        }

        if regime_labels is not None:
            common = r.index.intersection(regime_labels.index)
            known  = regime_labels.dropna().unique()
            ordered = ([rr for rr in _REGIME_ORDER if rr in known] +
                       [rr for rr in known if rr not in _REGIME_ORDER])
            for regime in ordered:
                mask  = regime_labels.loc[common] == regime
                r_reg = r.loc[common][mask]
                if len(r_reg) > 5:
                    rsh = (r_reg.mean() * td) / (r_reg.std() * np.sqrt(td) + 1e-8)
                    row[f'Sharpe ({regime})'] = round(rsh, 3)
                else:
                    row[f'Sharpe ({regime})'] = np.nan

        rows.append(pd.Series(row, name=col))

    return pd.DataFrame(rows)
