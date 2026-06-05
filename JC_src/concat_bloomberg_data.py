from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BLOOMBERG_DATA_DIR = PROJECT_ROOT / "Current_Data_Sources" / "Bloomberg_Data"
DEFAULT_OUTPUT_FILE = BLOOMBERG_DATA_DIR / "Concat_Data"


def ticker_from_filename(file_path: Path) -> str:
    """Extract the ticker prefix from a Bloomberg filename."""
    stem = file_path.stem
    if "_" not in stem:
        raise ValueError(f"Bloomberg file name does not contain an underscore: {file_path.name}")
    ticker = stem.split("_", 1)[0].strip()
    if not ticker:
        raise ValueError(f"Could not extract a ticker prefix from: {file_path.name}")
    return ticker


def load_bloomberg_series(file_path: Path) -> pd.Series:
    """Load a Bloomberg workbook as a date-indexed price series."""
    frame = pd.read_excel(file_path)
    frame = frame.dropna(how="all")
    if frame.empty:
        raise ValueError(f"No data found in workbook: {file_path.name}")

    normalized_columns = {column: str(column).strip().lower() for column in frame.columns}
    date_candidates = [column for column, normalized in normalized_columns.items() if "date" in normalized]
    date_column = date_candidates[0] if date_candidates else frame.columns[0]

    value_columns = [column for column in frame.columns if column != date_column]
    if not value_columns:
        raise ValueError(f"No price column found in workbook: {file_path.name}")

    price_candidates = [
        column for column in value_columns if "price" in normalized_columns.get(column, "")
    ]
    price_column = price_candidates[0] if price_candidates else value_columns[0]

    series_frame = frame[[date_column, price_column]].copy()
    series_frame[date_column] = pd.to_datetime(series_frame[date_column], errors="coerce")
    series_frame[price_column] = pd.to_numeric(series_frame[price_column], errors="coerce")
    series_frame = series_frame.dropna(subset=[date_column, price_column])

    if series_frame.empty:
        raise ValueError(f"No valid date/price rows found in workbook: {file_path.name}")

    series_frame = series_frame.sort_values(date_column)
    series_frame = series_frame.drop_duplicates(subset=[date_column], keep="last")

    ticker = ticker_from_filename(file_path)
    series = series_frame.set_index(date_column)[price_column].rename(ticker)
    series.index.name = "Date"
    return series


def iter_bloomberg_workbooks(data_dir: Path) -> list[Path]:
    """Return Bloomberg workbooks to concatenate, excluding output artifacts."""
    excluded_names = {"EDA.ipynb", "Overview", "Concat_Data"}
    workbooks: list[Path] = []

    for file_path in sorted(data_dir.rglob("*.xlsx")):
        if any(part in excluded_names for part in file_path.parts):
            continue
        workbooks.append(file_path)

    if not workbooks:
        raise FileNotFoundError(f"No Bloomberg workbooks found in {data_dir}")

    return workbooks


def concat_bloomberg_data(data_dir: Path = BLOOMBERG_DATA_DIR) -> pd.DataFrame:
    """Concatenate Bloomberg workbooks into a single date-indexed DataFrame."""
    series_list = []
    seen_tickers: set[str] = set()

    for file_path in iter_bloomberg_workbooks(data_dir):
        ticker = ticker_from_filename(file_path)
        if ticker in seen_tickers:
            raise ValueError(f"Duplicate ticker prefix detected: {ticker}")
        seen_tickers.add(ticker)
        series = load_bloomberg_series(file_path)
        series_list.append(series)

    combined = pd.concat(series_list, axis=1, join="outer")
    combined = combined.sort_index()
    combined = combined.loc[~combined.index.duplicated(keep="last")]
    combined.index.name = "Date"
    return combined


def write_concatenated_data(
    data_dir: Path = BLOOMBERG_DATA_DIR,
    output_file: Path = DEFAULT_OUTPUT_FILE,
) -> pd.DataFrame:
    """Build the Bloomberg dataset and write it to CSV."""
    combined = concat_bloomberg_data(data_dir)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_file, index=True, index_label="Date")
    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Concatenate Bloomberg workbook prices into one CSV.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=BLOOMBERG_DATA_DIR,
        help="Directory containing Bloomberg .xlsx workbooks.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="CSV file to write the concatenated prices to.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    combined = write_concatenated_data(args.data_dir, args.output_file)
    print(f"Wrote {combined.shape[0]} rows and {combined.shape[1]} columns to {args.output_file}")


if __name__ == "__main__":
    main()