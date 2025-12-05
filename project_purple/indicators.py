# project_purple/indicators.py

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    """
    Exponential moving average for a price series.
    """
    return series.ewm(span=period, adjust=False).mean()


def add_ema(df: pd.DataFrame, period: int, price_col: str = "close") -> pd.DataFrame:
    """
    Add an EMA column to a price DataFrame.

    Example:
        df = add_ema(df, 20)
        -> adds column 'ema_20'
    """
    col_name = f"ema_{period}"
    df[col_name] = ema(df[price_col], period)
    return df


def true_range(df: pd.DataFrame) -> pd.Series:
    """
    Compute True Range.

    TR_t = max(
        high_t - low_t,
        abs(high_t - close_{t-1}),
        abs(low_t - close_{t-1})
    )
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range over a given period.
    Uses a simple moving average of True Range.
    """
    tr = true_range(df)
    return tr.rolling(window=period, min_periods=1).mean()


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Add an ATR column to the DataFrame.

    Example:
        df = add_atr(df, 14)
        -> adds column 'atr_14'
    """
    col_name = f"atr_{period}"
    df[col_name] = atr(df, period)
    return df


def add_basic_trend_indicators(
    df: pd.DataFrame,
    ema_periods: Iterable[int] = (20, 50),
    atr_period: int = 14,
    breakout_lookback: int = 20,
) -> pd.DataFrame:
    """
    Convenience function to add the core indicators we care about for
    Project Purple:

    - EMA 20, EMA 50
    - ATR(14)
    - rolling 20-day high (for breakout logic)
    """
    # EMAs
    for p in ema_periods:
        df = add_ema(df, p)

    # ATR
    df = add_atr(df, atr_period)

    # 20-day high breakout level (exclude today by default)
    breakout_col = f"high_{breakout_lookback}d"
    df[breakout_col] = df["high"].rolling(window=breakout_lookback, min_periods=1).max()

    return df
