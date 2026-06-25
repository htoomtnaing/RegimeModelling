# Regime Modelling Submission

## Project Purpose

This directory contains a 3-stage regime-modelling workflow plus one integrated convenience runner.

The original stage files are:

1. `01_regime_analysis.py`
2. `02_regime_metrics_states.py`
3. `03_ETF_SAA_DR_RSDR_CVaR_Analysis_formula_explained.py`

For convenience, `integrated_regime_pipeline.py` combines the 3 stages into one run.

This README is written for both human reviewers and future LLM/code-agent development. It uses exact file names, explicit paths, and short section headings so the repository is easy to parse.

## Canonical Files

These are the submission-facing workflow files in the current directory:

- `01_regime_analysis.py`
- `01_regime_analysis.ipynb`
- `02_regime_metrics_states.py`
- `02_regime_metrics_states.ipynb`
- `03_ETF_SAA_DR_RSDR_CVaR_Analysis_formula_explained.py`
- `03_ETF_SAA_DR_RSDR_CVaR_Analysis_formula_explained.ipynb`
- `integrated_regime_pipeline.py`
- `integrated_regime_pipeline.ipynb`
- `requirements.txt`
- `README.md`

## Directory Map

Submission-relevant top-level structure:

```text
Data/
outputs/
src/
01_regime_analysis.py
01_regime_analysis.ipynb
02_regime_metrics_states.py
02_regime_metrics_states.ipynb
03_ETF_SAA_DR_RSDR_CVaR_Analysis_formula_explained.py
03_ETF_SAA_DR_RSDR_CVaR_Analysis_formula_explained.ipynb
integrated_regime_pipeline.py
integrated_regime_pipeline.ipynb
requirements.txt
README.md
```

Ignore files listed in `.gitignore` and the handoff file `context_handover.json` for submission and development planning.

## Inputs

All data used by the workflow comes from the `Data\` folder.

Canonical input locations include:

- `Data\fredmd_current.csv`
- `Data\nber_usrec.csv`
- `Data\altdata_monthly.csv`
- `Data\altdata_tcodes.csv`
- `Data\etf_returns.csv`
- `Data\Bloomberg_Data\`
- `Data\Farma_French\`
- `Data\Macro\`

## Outputs

All generated results go to the `outputs\` folder.

Integrated outputs are written to:

- `outputs\integrated\root_gmm\`
- `outputs\integrated\regime_states\`
- `outputs\integrated\cvar\`

Standalone stage outputs are also stored under `outputs\` when the stage files are run directly.

Current `outputs\` tree snapshot:

```text
outputs/
  CVaR/
    etf_cvar_gamma_tuning/
  GMM/
    figures/
  HMM/
  integrated/
    root_gmm/
      figures/
    regime_states/
    cvar/
```

## Run Order

Run the files in this order if you want the staged workflow:

1. `01_regime_analysis.py`
2. `02_regime_metrics_states.py`
3. `03_ETF_SAA_DR_RSDR_CVaR_Analysis_formula_explained.py`

If you want one combined entrypoint, run `integrated_regime_pipeline.py`.

## How To Run

From the repository root:

```bash
pip install -r requirements.txt
python 01_regime_analysis.py
python 02_regime_metrics_states.py
python 03_ETF_SAA_DR_RSDR_CVaR_Analysis_formula_explained.py
```

`requirements.txt` includes the runtime stack plus the notebook and export helpers used by the workflow, including the `duckdb` fallback used by the CVaR script.

Or run the integrated pipeline:

```bash
python integrated_regime_pipeline.py
```

`integrated_regime_pipeline.py` is kept in Jupytext percent format, so it can be exported back to an `.ipynb` notebook if needed.

## Notes for Future Development

- Treat `Data\` as the canonical input root.
- Treat `outputs\` as the canonical output root.
- Prefer the root stage files and `integrated_regime_pipeline.py` as the active workflow entrypoints.
- Use `src\` for shared helper logic.
- Keep the README and code paths explicit so future LLMs can reconstruct the pipeline without needing hidden context.
- Do not rely on ignored files or `context_handover.json` for the submission workflow.

## Workflow Summary

The integrated pipeline follows this sequence:

1. Daily regime analysis and GMM fitting.
2. Monthly regime metrics and state comparison.
3. ETF SAA, DR-CVaR, RS-CVaR, and RSDR-CVaR evaluation.

Each stage is represented by its own script, and the integrated script is only a convenience wrapper around those 3 connected runs.
