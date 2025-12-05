# project_purple/data_loader.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import pandas as pd

from project_purple.config import config
from project_purple.ib_client import IBClient


# Where the bars come from
BarSource = Literal["csv", "ib"]


@dataclass
class PriceBar:
    """
    Simple structure representing a single OHLCV bar.
    We mostly use DataFrames, but this documents the schema.
    """
    symbol: str
    date: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float


def _standardize_df(df: pd.DataFrame, symbol: str | None = None) -> pd.DataFrame:
    """
    Ensure we always get the same columns and index format.

    Final format:
        index: DatetimeIndex named 'date'
        columns: symbol, open, high, low, close, volume
    """

    # If there is an explicit date column, use it as index
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
    elif "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        df.index.name = "date"
    else:
        # assume the index is already a date-like index
        df.index = pd.to_datetime(df.index)
        if df.index.name is None:
            df.index.name = "date"

    # Attach symbol if provided
    if symbol is not None:
        df["symbol"] = symbol

    # Normalize column names to lowercase
    rename_map: dict[str, str] = {}
    for c in df.columns:
        lc = c.lower()
        if lc in {"open", "high", "low", "close", "volume", "symbol"}:
            rename_map[c] = lc

    if rename_map:
        df.rename(columns=rename_map, inplace=True)

    expected_cols = ["symbol", "open", "high", "low", "close", "volume"]
    cols_present = [c for c in expected_cols if c in df.columns]
    df = df[cols_present]

    df.sort_index(inplace=True)

    return df


def load_from_csv(
    symbol: str,
    file_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load historical data for a single symbol from a CSV file in /data.

    Default filename pattern:
        {symbol}_daily_60d.csv
    Ex: AAPL_daily_60d.csv, NVDA_daily_60d.csv

    CSV is expected to have at least:
        Date (or date) as first column, and Open, High, Low, Close, Volume.
    """
    if file_path is None:
        file_path = config.data_dir / f"{symbol}_daily_60d.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found for {symbol}: {file_path}")

    # Read CSV assuming first column is the date
    df = pd.read_csv(file_path, parse_dates=[0], index_col=0)

    if df.index.name is None:
        df.index.name = "date"

    df = _standardize_df(df, symbol=symbol)
    return df


def load_from_ib(
    symbol: str,
    ib_client: Optional[IBClient] = None,
    duration: str = "60 D",
) -> pd.DataFrame:
    """
    Load historical daily data from Interactive Brokers using IBClient.
    """
    client = ib_client or IBClient()
    df = client.get_daily_history(symbol, duration=duration)
    if df.empty:
        return df
    df = _standardize_df(df, symbol=symbol)
    return df


def load_history(
    symbol: str,
    source: BarSource = "csv",
    ib_client: Optional[IBClient] = None,
    duration: str = "60 D",
) -> pd.DataFrame:
    """
    Public entry point for the rest of the system.

    Examples:
        df = load_history("AAPL", source="csv")
        df = load_history("AAPL", source="ib", ib_client=IBClient())
    """
    if source == "csv":
        return load_from_csv(symbol)
    elif source == "ib":
        return load_from_ib(symbol, ib_client=ib_client, duration=duration)
    else:
        raise ValueError(f"Unknown data source: {source}")
