"""
portfolio_optimiser.py  —  Regime-conditional CVaR / DRO  (long-short)
=======================================================================

Design philosophy
-----------------
Position sizing is determined entirely by the CVaR / DRO mathematics.
No per-asset weight floors or ceilings are imposed — those would override
the risk signal the optimiser is computing.

The only portfolio-level constraint is:

    sum(w_i)        = 1          (net fully invested)
    sum(|w_i|)     <= gross_limit  (total leverage cap, default 1.6x)

The gross cap is itself a risk-budget decision: at 1.6x gross, the portfolio
can be, say, 130% long / 30% short.  It is NOT an arbitrary per-asset limit.

Why no per-asset bounds?
------------------------
CVaR/DRO are tail-risk optimisers.  In a Crisis regime the model correctly
wants to be, e.g., 100%+ long Rates (which rally in flight-to-quality) and
short equities (which sell off in the tail).  A cap of +60% on Rates would
suppress exactly the hedge the model is trying to build.  The gross_limit
ensures total leverage remains sensible without second-guessing individual
position signals.

Regularisation (lam)
--------------------
Pure CVaR (lam=0) is theoretically correct but can produce highly
concentrated solutions because it only cares about the worst (1-alpha)
tail.  A small L2 penalty toward 1/N (lam=0.05) improves out-of-sample
stability without materially distorting the tail-risk signal.  lam=0 is
supported for users who want the pure formulation.

Optimisers
----------
1. CVaR  — min  CVaR_alpha(w) + lam * ||w - 1/N||^2
            s.t. sum(w)=1,  sum(|w|) <= gross_limit
            Solved via scipy SLSQP.

2. DRO   — Wasserstein worst-case CVaR over ball of radius
            eps = kappa * sigma_hat / sqrt(S).
            The Wasserstein penalty eps*||w||_1 directly penalises gross
            exposure, reinforcing the leverage cap with distributional
            robustness.  Same weight solution as CVaR; CVaR bound shifted up.
"""

from __future__ import annotations

# Version marker — printed on import to confirm correct file is loaded
_VERSION = 'portfolio_optimiser v3 — long-short, no per-asset bounds'
print(f'[portfolio_optimiser] loaded: {_VERSION}')

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ── Pandas version compatibility ──────────────────────────────────────────────
def _compat_freq(freq: str) -> str:
    major, minor = (int(x) for x in pd.__version__.split(".")[:2])
    if (major, minor) >= (2, 2):
        return freq
    _MAP = {"ME": "M", "QE": "Q", "YE": "A", "BME": "BM", "BQE": "BQ", "BYE": "BA"}
    return _MAP.get(freq, freq)

# ── Asset universe ────────────────────────────────────────────────────────────
ASSET_TICKERS = [
    "MXCXDMHR", "RU30INTR", "MXEF",     "BCOMTR",
    "LGY7TRUH", "LUACTRUU", "LF98TRUU", "BCIT5T",
    "EMUSTRUU", "PUT",
]
ASSET_LABELS = {
    "MXCXDMHR": "Global Equity",  "RU30INTR": "US Equity",
    "MXEF":      "EM Equity",     "BCOMTR":   "Commodities",
    "LGY7TRUH":  "Rates 7-10yr",  "LUACTRUU": "US IG Credit",
    "LF98TRUU":  "US HY Credit",  "BCIT5T":   "US TIPS",
    "EMUSTRUU":  "EM Bonds",      "PUT":       "Short Vol",
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


# ── CVaR objective ────────────────────────────────────────────────────────────
def _cvar_obj(w, scenarios, alpha, lam):
    N      = len(w)
    losses = -scenarios @ w
    var    = np.percentile(losses, alpha * 100)
    tail   = losses[losses >= var]
    cvar   = float(np.mean(tail)) if len(tail) > 0 else var
    reg    = lam * float(np.sum((w - np.ones(N) / N) ** 2))
    return cvar + reg


# ── CVaR optimiser ────────────────────────────────────────────────────────────
def optimise_cvar(
    scenarios:   np.ndarray,       # (S, N) scenario return matrix
    alpha:       float = 0.95,     # CVaR confidence level
    lam:         float = 0.01,     # L2 regularisation toward 1/N (0 = pure CVaR)
    gross_limit: float = 1.60,     # sum(|w|) <= gross_limit; only portfolio constraint
    target_return: Optional[float] = None,
) -> OptimResult:
    """
    Minimise CVaR_alpha(w) + lam * ||w - 1/N||^2

    Constraints
    -----------
    sum(w)     = 1              net fully invested
    sum(|w|)  <= gross_limit    total leverage cap (the only position limit)

    No per-asset bounds.  The optimiser determines all position sizes from
    the tail-risk signal alone.
    """
    S, N = scenarios.shape
    w_eq = np.ones(N) / N

    constraints = [
        {"type": "eq",   "fun": lambda w: w.sum() - 1.0},
        {"type": "ineq", "fun": lambda w: gross_limit - np.sum(np.abs(w))},
    ]
    if target_return is not None:
        mu = scenarios.mean(axis=0)
        constraints.append(
            {"type": "ineq", "fun": lambda w: w @ mu - target_return / 252}
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = minimize(
            _cvar_obj, w_eq,
            args=(scenarios, alpha, lam),
            method="SLSQP",
            constraints=constraints,
            options={"maxiter": 2000, "ftol": 1e-12},
        )

    w_opt = res.x if res.success else w_eq
    # Renormalise net weight to exactly 1 (guard against numerical noise)
    w_opt = w_opt / w_opt.sum() if w_opt.sum() != 0 else w_eq

    losses   = -scenarios @ w_opt
    var_val  = np.percentile(losses, alpha * 100)
    tail     = losses[losses >= var_val]
    cvar_val = float(np.mean(tail)) if len(tail) > 0 else var_val

    return OptimResult(
        weights = w_opt,
        cvar    = cvar_val * np.sqrt(252),
        var     = var_val  * np.sqrt(252),
        status  = "optimal" if res.success else "fallback:equal_weight",
    )


# ── DRO (Wasserstein) ─────────────────────────────────────────────────────────
def optimise_dro(
    scenarios:   np.ndarray,
    alpha:       float = 0.95,
    kappa:       float = 1.0,      # Wasserstein radius multiplier
    lam:         float = 0.01,
    gross_limit: float = 1.60,
    target_return: Optional[float] = None,
) -> OptimResult:
    """
    Wasserstein DRO: worst-case CVaR over ambiguity ball of radius
        eps = kappa * sigma_hat / sqrt(S)

    The DRO penalty eps * ||w||_1 = eps * gross_exposure directly penalises
    leverage, adding distributional robustness on top of the gross_limit hard cap.
    No per-asset bounds.
    """
    S, N    = scenarios.shape
    sigma   = float(np.std(scenarios))
    eps     = kappa * sigma / np.sqrt(S)

    def dro_obj(w):
        return _cvar_obj(w, scenarios, alpha, lam) + eps * float(np.sum(np.abs(w)))

    constraints = [
        {"type": "eq",   "fun": lambda w: w.sum() - 1.0},
        {"type": "ineq", "fun": lambda w: gross_limit - np.sum(np.abs(w))},
    ]
    if target_return is not None:
        mu = scenarios.mean(axis=0)
        constraints.append(
            {"type": "ineq", "fun": lambda w: w @ mu - target_return / 252}
        )

    w_eq = np.ones(N) / N
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = minimize(
            dro_obj, w_eq,
            method="SLSQP",
            constraints=constraints,
            options={"maxiter": 2000, "ftol": 1e-12},
        )

    w_opt = res.x if res.success else w_eq
    w_opt = w_opt / w_opt.sum() if w_opt.sum() != 0 else w_eq

    losses   = -scenarios @ w_opt
    var_val  = np.percentile(losses, alpha * 100)
    tail     = losses[losses >= var_val]
    cvar_val = float(np.mean(tail)) if len(tail) > 0 else var_val

    return OptimResult(
        weights = w_opt,
        cvar    = cvar_val * np.sqrt(252) + eps * np.sqrt(252),
        var     = var_val  * np.sqrt(252),
        status  = ("optimal" if res.success else "fallback") + ":dro",
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
    wts    = np.where(lbl.values == current_regime, regime_boost, 1.0)
    wts    = wts / wts.sum()
    idx    = rng.choice(len(ret), size=n_scenarios, replace=True, p=wts)
    return ret.values[idx]


# ── Benchmark strategies (long-only for fair comparison) ─────────────────────
def weights_equal_weight(n: int) -> np.ndarray:
    return np.ones(n) / n

def weights_60_40(asset_names: list[str]) -> np.ndarray:
    w    = np.zeros(len(asset_names))
    eq   = next((t for t in ["MXCXDMHR","RU30INTR","MXEF"] if t in asset_names), None)
    bond = next((t for t in ["LGY7TRUH","LUACTRUU"]         if t in asset_names), None)
    if eq:   w[asset_names.index(eq)]   = 0.60
    if bond: w[asset_names.index(bond)] = 0.40
    return w / w.sum() if w.sum() > 0 else weights_equal_weight(len(asset_names))

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
    res = minimize(neg_sharpe, np.ones(N)/N, method="SLSQP",
                   bounds=[(0, 1)]*N,
                   constraints=[{"type":"eq","fun":lambda w: w.sum()-1}],
                   options={"maxiter":500,"ftol":1e-9})
    w = np.clip(res.x if res.success else np.ones(N)/N, 0, 1)
    return w / w.sum()

def weights_unconditional_cvar(
    returns_window: pd.DataFrame,
    alpha:       float = 0.95,
    lam:         float = 0.01,
    gross_limit: float = 1.60,
) -> np.ndarray:
    return optimise_cvar(
        returns_window.values, alpha=alpha, lam=lam, gross_limit=gross_limit
    ).weights


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
    # Optimisation
    alpha:              float = 0.95,
    lam:                float = 0.01,
    gross_limit:        float = 1.60,
    dro_kappa:          float = 1.0,
    # Execution
    rebalance_freq:     str   = "D",
    lookback_days:      int   = 756,
    n_scenarios:        int   = 500,
    regime_boost:       float = 3.0,    # scenario weight multiplier for current regime
    random_state:       int   = 42,
    # Expanding window
    expanding_window:   bool  = False,
    factor_matrix_full: Optional[pd.DataFrame] = None,
    n_components:       int   = 4,
    verbose:            bool  = True,
) -> BacktestResult:
    """
    Walk-forward backtest.  CVaR/DRO are long-short with no per-asset
    weight limits — only net=1 and gross<=gross_limit are enforced.
    Benchmarks are long-only for fair comparison.
    """
    assets = list(returns_test.columns)
    N      = len(assets)
    daily  = rebalance_freq == "D"

    rebal_dates = (
        set(returns_test.index)
        if daily else
        set(returns_test.resample(_compat_freq(rebalance_freq)).last().index)
    )

    strategies = [
        "CVaR (Regime)", "DRO (Regime)", "CVaR (Unconditional)",
        "Equal Weight",  "Risk Parity",  "MVO",
    ]
    port_rets   = {s: [] for s in strategies}
    wts_history = {s: [] for s in strategies}
    date_index  = []
    current_weights = {s: np.ones(N) / N for s in strategies}

    _last_refit_month    = None
    _cached_lbl_train    = hard_labels_train.copy()

    for date in returns_test.index:
        r = returns_test.loc[date].values
        for s in strategies:
            port_rets[s].append(current_weights[s] @ r)
        date_index.append(date)

        if date not in rebal_dates:
            continue

        # ── Current regime ─────────────────────────────────────────────────
        if date in regime_probs_test.index:
            current_regime = int(
                regime_probs_test.loc[date].idxmax().split("_")[1]
            )
        else:
            current_regime = 0

        # ── Expanding-window refit (monthly) ──────────────────────────────
        if expanding_window and factor_matrix_full is not None:
            mk = (date.year, date.month)
            if mk != _last_refit_month:
                _last_refit_month = mk
                try:
                    m = _refit_regime_model(
                        factor_matrix_full, date, n_components
                    )
                    _cached_lbl_train = m.hard_labels
                except Exception as e:
                    if verbose:
                        print(f"  [expanding] refit failed {date.date()}: {e}")

        # ── Scenario window ────────────────────────────────────────────────
        hist_ret = pd.concat([
            returns_train,
            returns_test.loc[returns_test.index < date],
        ]).tail(lookback_days)[assets]

        hist_lbl = pd.concat([
            _cached_lbl_train,
            hard_labels_test.loc[hard_labels_test.index < date],
        ]).tail(lookback_days)

        # ── CVaR (regime-conditional, long-short, no per-asset bounds) ────
        scen = build_regime_scenarios(
            hist_ret, hist_lbl, current_regime,
            n_scenarios=n_scenarios, lookback_days=lookback_days,
            regime_boost=regime_boost, random_state=random_state,
        )
        res_cvar = optimise_cvar(scen, alpha=alpha, lam=lam,
                                  gross_limit=gross_limit)
        res_cvar.asset_names = assets
        current_weights["CVaR (Regime)"] = res_cvar.weights

        # ── DRO (regime-conditional, long-short, no per-asset bounds) ─────
        res_dro = optimise_dro(scen, alpha=alpha, kappa=dro_kappa, lam=lam,
                                gross_limit=gross_limit)
        res_dro.asset_names = assets
        current_weights["DRO (Regime)"] = res_dro.weights

        # ── CVaR unconditional (long-short, no per-asset bounds) ──────────
        current_weights["CVaR (Unconditional)"] = weights_unconditional_cvar(
            hist_ret, alpha=alpha, lam=lam, gross_limit=gross_limit
        )

        # ── Benchmarks (long-only) ─────────────────────────────────────────
        current_weights["Equal Weight"] = weights_equal_weight(N)
        current_weights["Risk Parity"]  = weights_risk_parity(hist_ret.tail(252))
        current_weights["MVO"]          = weights_mvo(hist_ret.tail(252))

        for s in strategies:
            wts_history[s].append(
                pd.Series(current_weights[s], index=assets, name=date)
            )

        if verbose and not daily:
            rn     = regime_names.get(current_regime, f"R{current_regime}")
            w      = pd.Series(res_cvar.weights, index=assets)
            longs  = w[w >  0.01].nlargest(3)
            shorts = w[w < -0.01]
            ls_str = (
                "L: " + ", ".join(
                    f"{ASSET_LABELS.get(a,a)}={v:.0%}" for a, v in longs.items()
                ) + (
                    " | S: " + ", ".join(
                        f"{ASSET_LABELS.get(a,a)}={v:.0%}" for a, v in shorts.items()
                    ) if len(shorts) > 0 else ""
                )
            )
            print(f"  {date.date()}  [{rn:15s}]  {ls_str}")
        elif verbose and daily and date.day == 1:
            rn      = regime_names.get(current_regime, f"R{current_regime}")
            n_short = (res_cvar.weights < -0.01).sum()
            gross   = np.abs(res_cvar.weights).sum()
            print(f"  {date.date()}  [{rn:15s}]  "
                  f"shorts={n_short}  gross={gross:.2f}x")

    # ── Assemble ──────────────────────────────────────────────────────────────
    port_ret_df = pd.DataFrame(port_rets, index=date_index)
    wts_hist_df = {
        s: pd.DataFrame(wts_history[s]) for s in strategies if wts_history[s]
    }
    regime_labels = hard_labels_test.map(regime_names)
    metrics = compute_metrics(port_ret_df, alpha=alpha,
                              regime_labels=regime_labels)
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
    for col in port_returns.columns:
        r        = port_returns[col].dropna()
        ann_ret  = r.mean() * td
        ann_vol  = r.std()  * np.sqrt(td)
        sharpe   = ann_ret / (ann_vol + 1e-8)

        # Sortino: downside deviation below zero
        down    = (r - rf_daily)[r < rf_daily]
        d_std   = np.sqrt((down**2).mean()) * np.sqrt(td) if len(down) > 0 else 1e-8
        sortino = ann_ret / (d_std + 1e-8)

        # Historical CVaR
        s_r     = np.sort(r.values)
        var_idx = max(int(np.floor((1 - alpha) * len(s_r))), 1)
        cvar_v  = -s_r[:var_idx].mean() * np.sqrt(td)

        # Drawdown
        cum    = (1 + r).cumprod()
        dd     = (cum - cum.cummax()) / cum.cummax()
        max_dd = dd.min()
        calmar = ann_ret / (-max_dd + 1e-8)

        row = {
            "Ann. Return (%)":             round(ann_ret * 100, 2),
            "Ann. Vol (%)":                round(ann_vol * 100, 2),
            "Sharpe Ratio":                round(sharpe,        3),
            "Sortino Ratio":               round(sortino,       3),
            f"CVaR {int(alpha*100)}% (%)": round(cvar_v  * 100, 2),
            "Max Drawdown (%)":            round(max_dd  * 100, 2),
            "Calmar Ratio":                round(calmar,        3),
        }

        if regime_labels is not None:
            common = r.index.intersection(regime_labels.index)
            # Use ALL known regimes (not just ones in common) so every regime
            # always has a column even if only a few days survive alignment
            all_regimes = sorted(regime_labels.dropna().unique())
            for regime in all_regimes:
                mask  = regime_labels.loc[common] == regime
                r_reg = r.loc[common][mask]
                if len(r_reg) > 5:   # lowered from 20 to catch small regimes
                    rsh = (r_reg.mean() * td) / (r_reg.std() * np.sqrt(td) + 1e-8)
                    row[f"Sharpe ({regime})"] = round(rsh, 3)
                else:
                    row[f"Sharpe ({regime})"] = np.nan  # always include column

        rows.append(pd.Series(row, name=col))

    return pd.DataFrame(rows)
