# project_purple/data_loader.py

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

# ----------------------------------------------------------------------
# Resolve correct data directory: project_purple/data
# ----------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent          # inner project_purple
DATA_DIR = ROOT_DIR / "data"


def _parse_date_column(series: pd.Series) -> pd.Series:
    """
    Heuristically detect the date format for a column and parse it.

    Handles:
      - '12/7/2020' style (mm/dd/YYYY)
      - '2020-12-07' style (YYYY-mm-dd)
    And ignores junk rows like 'Ticker' or 'date'.
    """
    s = series.astype(str).str.strip()

    sample: Optional[str] = None
    for v in s:
        if not v:
            continue
        lower = v.lower()
        if lower in ("ticker", "date"):
            continue
        sample = v
        break

    fmt: Optional[str] = None
    if sample is not None:
        if "/" in sample:
            # e.g. 12/7/2020
            fmt = "%m/%d/%Y"
        elif "-" in sample:
            # e.g. 2020-12-07
            fmt = "%Y-%m-%d"

    if fmt is not None:
        # Use a specific format => no warnings, faster
        return pd.to_datetime(s, format=fmt, errors="coerce")
    else:
        # Fallback: let pandas/dateutil figure it out
        return pd.to_datetime(s, errors="coerce")


def load_from_csv(symbol: str, limit_days: Optional[int] = None) -> pd.DataFrame:
    """
    Load OHLCV data for a symbol from data/{SYMBOL}_daily.csv.

    Handles BOTH:
      1. Clean files created by data_downloader.py:
         date,symbol,open,high,low,close,volume
      2. Yahoo-style messy files where:
         - the first column holds dates
         - first few rows may be 'Ticker', 'date', etc.

    Returns a standardized DataFrame with:
        index   = DatetimeIndex named 'date'
        columns = ['symbol', 'open', 'high', 'low', 'close', 'volume']
    """
    path = DATA_DIR / f"{symbol}_daily.csv"

    if not path.exists():
        raise FileNotFoundError(f"CSV file not found for {symbol}: {path}")

    # Read raw CSV with no date parsing
    df = pd.read_csv(path)

    # --------------------------------------------------
    # 1) Decide which column is the date column
    # --------------------------------------------------
    if "date" in df.columns:
        # Clean format: explicit 'date' column
        date_col = "date"
    else:
        # Yahoo-style: first column holds date-ish values
        date_col = df.columns[0]

    # --------------------------------------------------
    # 2) Parse dates with our heuristic
    # --------------------------------------------------
    df[date_col] = _parse_date_column(df[date_col])

    # --------------------------------------------------
    # 3) Drop rows where date could not be parsed
    #    (this removes 'Ticker', 'date', etc.)
    # --------------------------------------------------
    df = df[~df[date_col].isna()]

    if df.empty:
        raise ValueError(f"No valid rows after parsing dates in {path}")

    # --------------------------------------------------
    # 4) Set datetime index
    # --------------------------------------------------
    df.set_index(date_col, inplace=True)
    df.index.name = "date"
    df.sort_index(inplace=True)

    # --------------------------------------------------
    # 5) Ensure numeric types for OHLCV
    # --------------------------------------------------
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Optional: trim to last N days
    if limit_days is not None:
        df = df.tail(limit_days)

    # --------------------------------------------------
    # 6) Validate required columns
    # --------------------------------------------------
    expected_cols = ["symbol", "open", "high", "low", "close", "volume"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    # Return in consistent column order
    return df[expected_cols]


def load_history(
    symbol: str,
    source: str = "csv",
    limit_days: Optional[int] = None,
) -> pd.DataFrame:
    """
    Unified history loader.

    Currently supports:
      - source="csv": load from local data/{SYMBOL}_daily.csv

    (Later we can add 'ib' to load from Interactive Brokers.)
    """
    source = source.lower().strip()

    if source == "csv":
        return load_from_csv(symbol, limit_days=limit_days)

    raise NotImplementedError(f"Unknown data source: {source}")
