# project_purple/data_downloader.py

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_ticker(ticker: str, period: str = "5y") -> None:
    """
    Download historical daily OHLCV data for a ticker using yfinance
    and save as data/{TICKER}_daily.csv in a standardized format:
        date,symbol,open,high,low,close,volume
    """
    print(f"Downloading {ticker}...")

    df = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        print(f"ERROR: No data returned for {ticker}")
        return

    # Keep only needed columns
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

    # Standardize column names
    df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        },
        inplace=True,
    )

    # Add symbol column
    df["symbol"] = ticker

    # Move symbol to first column
    df.index.name = "date"
    df = df[["symbol", "open", "high", "low", "close", "volume"]]

    out_path = DATA_DIR / f"{ticker}_daily.csv"
    # Explicitly write index label 'date'
    df.to_csv(out_path, index_label="date")

    print(f"Saved {ticker} to {out_path}")


def download_all() -> None:
    tickers = ["AAPL", "NVDA", "SPY"]
    for t in tickers:
        download_ticker(t)


if __name__ == "__main__":
    download_all()
