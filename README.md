# GMM vs HMM Market Regime Detection

This project uses macro factor ETFs to compare Gaussian Mixture Models (GMM) and Hidden Markov Models (HMM) for market regime detection.

The main notebook uses the BIC-selected 3-regime setup. A 4-regime notebook is kept as an alternative specification.

## Setup

```bash
pip install -r requirements.txt
```

## Run

Main 3-regime analysis:

```bash
jupyter notebook notebooks/market_regime_GMMvsHMM.ipynb
```

Optional 4-regime version:

```bash
jupyter notebook notebooks/market_regime_GMMvsHMM_4regime.ipynb
```

## Notes

- Data is downloaded with `yfinance`.
- Models use standardized daily log returns.
- Charts are shown in the notebooks, not saved automatically.
- Regime numbers are model labels and should be interpreted using the summary tables.
- This is for learning and analysis, not investment advice.
