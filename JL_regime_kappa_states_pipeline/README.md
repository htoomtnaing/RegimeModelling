# Macro-regime detection — Cohen's κ, regime states & a final HMM

A self-contained pipeline that scores how well unsupervised regime detectors line up with
NBER recessions, visualises the regime paths, reports stability/fit diagnostics, and then
builds and characterises one chosen final model. Everything it needs lives in this folder —
the input CSVs in `data/`, the `regime_taa.py` library, and the notebook — so it runs with
no outside dependencies.

## Quick start (Windows)

```
setup.bat     :: create .venv and install requirements
run.bat       :: execute the notebook end-to-end (regenerates outputs\, a few minutes)
start.bat     :: open the notebook in Jupyter Lab to read / re-run interactively
```

On other platforms: create a virtual environment, `pip install -r requirements.txt`, then
either open `regime_kappa_states.ipynb` in Jupyter, or run it headless:

```
python build_nb_regime_kappa_states.py        # (re)generate the notebook from source
python run_notebook.py                         # execute in place with live per-stage progress
# (or: jupyter nbconvert --to notebook --execute --inplace regime_kappa_states.ipynb)
```

`run_notebook.py` streams each stage's progress to the console as it runs (the κ table per
config, then the diagnostics refits — the longest stage), so the run is never a silent black box.

> Built and tested on Python 3.14 (numpy 2.x / pandas 3.x). `build_nb_regime_kappa_states.py`
> is the notebook's source of truth — edit it and rerun to regenerate `regime_kappa_states.ipynb`.

## What the notebook does

1. **Cohen's κ table** — for five feature configurations (`macro+eng`, `comb_all/all`,
   `comb_all/bn_meanrank`, `altdata_full/all`, `altdata_full/bn_meanrank`) it fits two
   engines, a Gaussian mixture (**GMM**) and a Gaussian hidden Markov model (**HMM**), and
   scores each engine's crisis flag against NBER recessions with Cohen's κ. Each row reports
   the number of input **features** and the number of **principal components** the engine
   sees.
2. **Regime-state paths** — the month-by-month regime state for every config × engine, with
   NBER recessions shaded and the model's crisis state highlighted.
3. **Diagnostics** — cluster-stability (seed ARI), silhouette and BIC for every config × engine.
4. **Final model** — the HMM of `altdata_full/bn_meanrank` (almost the strongest κ with the
   fewest features), characterised by per-state feature distributions, PC factor loadings
   (with intuitive labels), per-state PC distributions, intuitive state labels, the
   transition matrix, and a per-month state-probability CSV.

## What Cohen's κ means

Cohen's kappa measures *chance-corrected agreement* between two 0/1 labelings of the same
months — here the model's binary crisis flag vs the NBER recession indicator:
`κ = (p_o − p_e) / (1 − p_e)`, where `p_o` is the fraction of months the two labels agree and
`p_e` is the agreement expected by chance given each label's base rate. **κ = 1** is perfect
agreement, **κ = 0** is no better than chance, **κ < 0** is worse than chance. Because
recessions are rare, κ — unlike raw accuracy — is not inflated by a model that trivially
calls every month "expansion".

**These κ are in-sample.** Each engine is fit on all available months and scored over the
same span, so it has seen the recessions it then flags — read the numbers as an *upper bound
on detectability*, not out-of-sample skill. Feature selection for the `bn_meanrank` rows is
look-ahead-free (it sees only data up to 2017-12).

## Inputs (`data/`)

| file | contents |
|---|---|
| `fredmd_current.csv` | FRED-MD monthly macro panel (raw levels + transform codes) |
| `nber_usrec.csv` | NBER recession indicator (monthly 0/1) — the κ target |
| `altdata_monthly.csv` | alternative-data universe (rates, factor returns, commodity/credit series) |
| `altdata_tcodes.csv` | per-column FRED transform codes for the alt-data |

## Outputs (`outputs/`)

| file | contents |
|---|---|
| `regime_kappa_table.csv` | the five-config κ table: `config, n_feat, n_PCs, n_rec, GMM, HMM` |
| `fig_state_paths.png` | regime-state paths for all configs × engines, NBER shaded |
| `table_diagnostics.csv` | silhouette, BIC and seed-ARI per config × engine (grouped by engine) |
| `final_feature_by_state.csv` | the 9 features per final-model HMM state (z, transformed & untransformed means + descriptions) |
| `final_pc_loadings.csv` | factor loadings of the 7 PCs over the 9 features (+ variance shares) |
| `final_pc_by_state.csv` | mean of each PC within each HMM state |
| `final_transition_matrix.csv` | the 6×6 HMM transition matrix (rows sum to 1) |
| `final_state_probabilities.csv` | per month: `date, state, p_state0..p_state5` |
| `fig_feature_by_state_z.png` | heatmap of standardized feature means by state |
| `fig_pc_loadings.png` | heatmap of the PC factor loadings |
| `fig_pc_by_state.png` | heatmap of mean PC scores by state |
| `fig_transition_matrix.png` | heatmap of the transition matrix |

## Reproducibility

Thread pinning (set before importing numpy) plus full-SVD PCA plus fixed seeds make every
number byte-reproducible across runs.
