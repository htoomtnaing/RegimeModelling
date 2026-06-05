# GMM vs HMM Market Regime Detection

This project uses macro factor ETFs to compare Gaussian Mixture Models (GMM) and Hidden Markov Models (HMM) for market regime detection.

The main notebook uses the BIC-selected 3-regime setup. A 4-regime notebook is kept as an alternative specification.

The repository also includes a Bloomberg data concatenation utility that combines the workbook files in `Current_Data_Sources/Bloomberg_Data` into a single date-indexed CSV.

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

Build the Bloomberg price panel:

```bash
python JC_Notebooks/src/concat_bloomberg_data.py
```

This reads every Bloomberg `.xlsx` file in `Current_Data_Sources/Bloomberg_Data`, labels each series with the text before the first underscore in the filename, and writes the combined CSV to `Current_Data_Sources/Bloomberg_Data/Concat_Data`.

## Notes

- Data is downloaded with `yfinance`.
- Bloomberg workbooks are concatenated locally with pandas and `openpyxl`.
- Models use standardized daily log returns.
- Charts are shown in the notebooks, not saved automatically.
- Regime numbers are model labels and should be interpreted using the summary tables.
- This is for learning and analysis, not investment advice.
