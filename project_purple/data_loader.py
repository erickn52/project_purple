from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


# Keep this list in sync with the project’s “known symbols” expectations.
# (Downloader uses scanner symbols; loader can still keep a stable subset.)
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
    Resolve the ONE canonical data directory.

    Canonical (Option A):
      - repo-root ./data

    Safety:
      - If repo-root ./project_purple/data exists, hard-fail.
        That folder previously caused corrupted OHLCV + path ambiguity.
    """
    repo_root = Path(__file__).resolve().parents[1]

    canonical = repo_root / "data"
    legacy_bad = repo_root / "project_purple" / "data"  # DO NOT USE

    # Hard gate: this folder must not exist (prevents silent reintroduction).
    if legacy_bad.exists():
        raise RuntimeError(
            "Unsafe data directory detected:\n"
            f"  {legacy_bad}\n\n"
            "Project Purple is locked to ONE canonical data folder:\n"
            f"  {canonical}\n\n"
            "Fix:\n"
            "  1) Delete the legacy folder:\n"
            f"     Remove-Item -Recurse -Force {legacy_bad}\n"
            "  2) Re-run the system.\n"
        )

    # Ensure canonical exists (but do not create legacy folders).
    canonical.mkdir(parents=True, exist_ok=True)
    return canonical


def _clean_daily_csv(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Normalize your CSVs into:
      date, symbol, open, high, low, close, volume

    Handles the "Price" date column, and filters out junk rows like:
      Ticker,<sym>,<sym>,...
      date,,,,,,
    """
    df = raw.copy()

    # Standardize date column name
    if "Price" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"Price": "date"})

    required = {"date", "symbol", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {symbol} CSV: {sorted(missing)}")

    # Remove obvious junk header rows that appear inside the file
    date_str = df["date"].astype(str).str.strip().str.lower()
    bad_tokens = {"ticker", "date", "", "nan", "none"}
    df = df[~date_str.isin(bad_tokens)].copy()

    # Parse dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()

    # Numeric conversion
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with missing price data
    df = df.dropna(subset=["open", "high", "low", "close"]).copy()

    # Ensure expected column order
    expected_cols = ["date", "symbol", "open", "high", "low", "close", "volume"]
    df = df.reindex(columns=expected_cols)

    # Sort + de-dup dates (keep last)
    df = (
        df.sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )

    return df


def load_symbol_daily(symbol: str) -> pd.DataFrame:
    """
    Load a single symbol's daily data from CSV.

    Canonical path only:
      - repo-root ./data/<SYMBOL>_daily.csv
    """
    symbol = symbol.upper().strip()

    data_dir = get_data_dir()
    csv_path = data_dir / f"{symbol}_daily.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV for {symbol} not found at: {csv_path}\n"
            f"Canonical data dir is: {data_dir}\n"
            "If you need to fetch data, run:\n"
            f"  python -u .\\project_purple\\data_downloader.py --symbols {symbol}\n"
        )

    raw = pd.read_csv(csv_path)
    df = _clean_daily_csv(raw=raw, symbol=symbol)
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
        resolved = get_data_dir()
        print("Resolved data dir:", resolved)

        spy_df = load_symbol_daily("SPY")
        print("\nSPY head:")
        print(spy_df.head())
        print(f"\nSPY rows: {len(spy_df)}")

    except Exception as e:
        print(f"Error while loading data: {e}")
