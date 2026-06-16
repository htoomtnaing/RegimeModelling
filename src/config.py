from pathlib import Path


TICKERS = {
    "Equity": "ACWI",
    "Rates": "IEF",
    "Credit": "LQD",
    "Commodities": "DBC",
    "EM": "EEM",
    "FX": "UUP",
    "ShortVol": "PUTW",
    "Inflation": "TIP",
}

START_DATE = "2010-01-01"
END_DATE = None


COVARIANCE_TYPE = "full"
RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
