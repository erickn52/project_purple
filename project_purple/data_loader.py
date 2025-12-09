from pathlib import Path
from typing import Dict

import pandas as pd


# Keep this list in sync with data_downloader.TICKERS
TICKERS = [
    "AAPL",
    "NVDA",
    "SPY",
    "TSLA",
    "AMD",
    "MSFT",
    "META",
    "AMZN",
    "IBKR",
    "AMDL",
]


def get_data_dir() -> Path:
    """
    Return the path to the data directory:
        .../project_purple/project_purple/data
    """
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "project_purple" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def load_symbol_daily(symbol: str) -> pd.DataFrame:
    """
    Load a single symbol's daily data from the CSV created by data_downloader.py.

    CSV structure:

        columns: Price, symbol, open, high, low, close, volume

        Row 0: Ticker,<sym>,<sym>,...,<sym>
        Row 1: date,,,,,,
        Row 2+: YYYY-MM-DD, <sym>, open, high, low, close, volume

    This function:
        - drops the first 2 header rows,
        - renames 'Price' to 'date',
        - parses 'date' as datetime,
        - ensures numeric types for OHLCV,
        - returns a clean DataFrame.
    """
    data_dir = get_data_dir()
    csv_path = data_dir / f"{symbol}_daily.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV for {symbol} not found at: {csv_path}")

    # Read entire CSV
    raw = pd.read_csv(csv_path)

    if raw.shape[0] < 3:
        raise ValueError(
            f"CSV for {symbol} looks too short (rows={raw.shape[0]}). "
            f"Expected at least 3 rows including headers."
        )

    # Drop the first two header rows (Ticker row + date row)
    df = raw.iloc[2:].copy()

    # Rename 'Price' -> 'date'
    if "Price" not in df.columns:
        raise ValueError(
            f"Expected 'Price' column in {symbol} CSV, got columns: {list(df.columns)}"
        )

    df = df.rename(columns={"Price": "date"})

    # Parse dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # Ensure column order
    expected_cols = ["date", "symbol", "open", "high", "low", "close", "volume"]
    df = df.reindex(columns=expected_cols)

    # Numeric conversion
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with missing price data
    df = df.dropna(subset=["open", "high", "low", "close"])

    # Symbol as string
    df["symbol"] = df["symbol"].astype(str)

    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)

    return df


def load_all_daily() -> Dict[str, pd.DataFrame]:
    """
    Load daily data for all symbols in TICKERS into a dict:
        { "AAPL": DataFrame, "NVDA": DataFrame, ... }
    """
    data: Dict[str, pd.DataFrame] = {}

    for symbol in TICKERS:
        try:
            df = load_symbol_daily(symbol)
            data[symbol] = df
        except Exception as e:
            print(f"WARNING: Failed to load {symbol}: {e}")

    return data


if __name__ == "__main__":
    # Simple diagnostics when you run this file directly
    print("Testing data_loader...")

    try:
        aapl_df = load_symbol_daily("AAPL")
        print("\nAAPL head:")
        print(aapl_df.head())

        print("\nAAPL tail:")
        print(aapl_df.tail())

        print(f"\nAAPL rows: {len(aapl_df)}")
    except Exception as e:
        print(f"Error while loading AAPL: {e}")
