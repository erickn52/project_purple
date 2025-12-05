"""
data_loader.py

Historical data access layer using IB (via ib_insync).

Responsibilities:
- Define contracts (e.g., US stocks)
- Request historical OHLCV bars from IB
- Convert to pandas DataFrame
- Save to CSV in the project's data folder
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from ib_insync import Stock, util  # type: ignore

from . import config
from .ib_client import ib_client


def make_stock_contract(symbol: str, exchange: str = "SMART", currency: str = "USD") -> Stock:
    """
    Create an IB Stock contract for a US-listed stock.

    Parameters
    ----------
    symbol : str
        Ticker symbol, e.g. 'AAPL'
    exchange : str
        Exchange route, usually 'SMART'
    currency : str
        Currency, usually 'USD'

    Returns
    -------
    Stock
        ib_insync Stock contract object
    """
    return Stock(symbol, exchange, currency)


def get_daily_bars(
    symbol: str,
    lookback_days: int = 90,
    use_rth: bool = True,
    what_to_show: str = "TRADES",
    save_csv: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Request historical daily bars from IB and optionally save to CSV.

    Parameters
    ----------
    symbol : str
        Ticker symbol, e.g. 'AAPL'
    lookback_days : int
        How many calendar days of history to request (IB limitation).
    use_rth : bool
        If True, only use Regular Trading Hours (no pre/post market).
    what_to_show : str
        IB 'whatToShow' parameter. 'TRADES' is usually appropriate.
    save_csv : bool
        If True, save the resulting DataFrame to a CSV under data/.

    Returns
    -------
    Optional[pd.DataFrame]
        DataFrame indexed by datetime with OHLCV columns.
        Returns None if the request fails or no data returned.
    """
    # Ensure IB is connected
    if not ib_client.is_connected():
        connected = ib_client.connect()
        if not connected:
            print("DataLoader: Could not connect to IB, aborting historical data request.")
            return None

    contract = make_stock_contract(symbol)

    duration_str = f"{lookback_days} D"  # IB format, e.g. '90 D'
    bar_size = "1 day"

    print(
        f"DataLoader: Requesting {duration_str} of {bar_size} bars for {symbol} "
        f"(RTH={use_rth}, whatToShow={what_to_show})..."
    )

    try:
        bars = ib_client.ib.reqHistoricalData(
            contract=contract,
            endDateTime="",           # '' means "up to now"
            durationStr=duration_str,
            barSizeSetting=bar_size,
            whatToShow=what_to_show,
            useRTH=use_rth,
            formatDate=1,             # human-readable datetimes
            keepUpToDate=False,
        )
    except Exception as exc:
        print(f"DataLoader: Error requesting historical data -> {exc}")
        return None

    if not bars:
        print("DataLoader: No bars returned from IB.")
        return None

    # Convert to DataFrame
    df = util.df(bars)

    # Normalize column names
    df.rename(
        columns={
            "date": "datetime",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
            "barCount": "bar_count",
            "average": "vwap",
        },
        inplace=True,
    )

    # Set datetime index
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)

    # Add simple dollar-volume column for liquidity checks
    df["dollar_volume"] = df["close"] * df["volume"]

    # Save to CSV if requested
    if save_csv:
        csv_path = _make_csv_path(symbol, lookback_days)
        df.to_csv(csv_path)
        print(f"DataLoader: Saved {len(df)} rows to {csv_path}")

    return df


def _make_csv_path(symbol: str, lookback_days: int) -> Path:
    """
    Build the path under the project's data folder for a symbol's daily bars.
    """
    file_name = f"{symbol.upper()}_daily_{lookback_days}d.csv"
    return config.DATA_PATH / file_name
