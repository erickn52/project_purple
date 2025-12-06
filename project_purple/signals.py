# project_purple/signals.py

from __future__ import annotations

import pandas as pd

from project_purple.settings import strategy_settings


def add_basic_long_signal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a basic long-entry signal based on:
      - Uptrend (EMA 20 above EMA 50, close above EMA 20)
      - 20-day breakout (close > prior 20-day high)
      - Volatility filter (avoid ATR spikes)
      - Optional market regime filter (SPY trend)
    """
    df = df.copy()

    # Uptrend conditions
    up_trend = (df["ema_20"] > df["ema_50"]) & (df["close"] > df["ema_20"])

    # Prior 20-day high (excluding today) for breakout condition
    prior_20d_high = df["high"].shift(1).rolling(window=20, min_periods=20).max()
    breakout = df["close"] > prior_20d_high

    # Volatility filter: avoid extreme ATR spikes
    max_ratio = strategy_settings.atr_vol_max_mult
    vol_ok = df["atr_vol_ratio"].notna() & (df["atr_vol_ratio"] <= max_ratio)

    # Optional market regime filter
    if "market_long_ok" in df.columns and strategy_settings.use_market_regime:
        regime_ok = df["market_long_ok"].fillna(False)
    else:
        regime_ok = True  # no regime info -> allow

    # Store filters for debugging/inspection
    df["vol_ok"] = vol_ok
    if isinstance(regime_ok, pd.Series):
        df["regime_ok"] = regime_ok
    else:
        df["regime_ok"] = True

    # Final signal
    df["long_signal"] = up_trend & breakout & vol_ok & df["regime_ok"]

    return df
