# project_purple/signals.py

from __future__ import annotations

import pandas as pd


def add_basic_long_signal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a basic long-entry signal based on:
      - Uptrend (EMA 20 above EMA 50, close above EMA 20)
      - 20-day breakout (close > prior 20-day high)

    Assumes df already has columns:
      - 'close'
      - 'ema_20'
      - 'ema_50'
      - 'high'
    """

    # Uptrend conditions
    up_trend = (df["ema_20"] > df["ema_50"]) & (df["close"] > df["ema_20"])

    # Prior 20-day high (excluding today)
    prior_20d_high = df["high"].shift(1).rolling(window=20, min_periods=20).max()

    breakout = df["close"] > prior_20d_high

    # Final long signal: uptrend + breakout
    df["long_signal"] = up_trend & breakout

    return df
