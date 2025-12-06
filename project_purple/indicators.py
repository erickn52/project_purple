# project_purple/indicators.py

from __future__ import annotations

import pandas as pd
from ta.volatility import AverageTrueRange

from project_purple.settings import strategy_settings


def add_basic_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add core indicators needed by the strategy:
      - ema_20, ema_50 (trend)
      - atr_14 (volatility)
      - high_20d (breakout level)
      - atr_slow, atr_vol_ratio (for volatility filter)
    """
    df = df.copy()

    # Trend EMAs
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()

    # ATR(14)
    atr_indicator = AverageTrueRange(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        window=14,
        fillna=False,
    )
    df["atr_14"] = atr_indicator.average_true_range()

    # 20-day high for breakout logic
    df["high_20d"] = df["high"].rolling(window=20, min_periods=20).max()

    # Slower ATR mean for volatility filter
    window = strategy_settings.atr_vol_window
    df["atr_slow"] = df["atr_14"].rolling(
        window=window,
        min_periods=max(10, window // 2),
    ).mean()

    # Ratio of fast ATR to slow ATR
    df["atr_vol_ratio"] = df["atr_14"] / df["atr_slow"]

    return df
