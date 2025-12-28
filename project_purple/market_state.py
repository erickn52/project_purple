# project_purple/market_state.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import sys

import numpy as np
import pandas as pd

# Internal imports: support package import (project_purple.market_state) and script runs.
try:
    from .data_loader import load_symbol_daily
except ImportError:  # pragma: no cover
    try:
        from data_loader import load_symbol_daily
    except ModuleNotFoundError:  # pragma: no cover
        from project_purple.data_loader import load_symbol_daily

# Centralized regime policy (single source of truth).
try:
    from .regime_risk import get_regime_policy
except ImportError:  # pragma: no cover
    try:
        from regime_risk import get_regime_policy
    except ModuleNotFoundError:  # pragma: no cover
        from project_purple.regime_risk import get_regime_policy

# Centralized strategy config (single source of truth for benchmark ticker).
try:
    from .config import strategy_config
except ImportError:  # pragma: no cover
    try:
        from config import strategy_config
    except ModuleNotFoundError:  # pragma: no cover
        from project_purple.config import strategy_config


# Basic parameters for regime detection
REGIME_FAST_MA = 50
REGIME_SLOW_MA = 200
REGIME_ATR_WINDOW = 14

MARKET_SYMBOL = str(strategy_config.benchmark_ticker).upper()  # e.g., SPY/QQQ/DIA

# Console colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


@dataclass
class MarketState:
    """Represents the high-level state of the overall market."""

    symbol: str
    as_of_date: pd.Timestamp
    regime: str  # "BULL", "BEAR", "CHOPPY"
    close: float
    ma_fast: float
    ma_slow: float
    atr: Optional[float]

    @property
    def trade_long(self) -> bool:
        """
        Whether long trades are allowed in this market regime.

        This uses regime_risk.get_regime_policy() as the single source of truth.
        NOTE: In this repo, RegimePolicy uses the field name `trade_long`.
        """
        policy = get_regime_policy(self.regime)
        return bool(policy.trade_long)

    @property
    def risk_multiplier(self) -> float:
        """
        Risk multiplier for position sizing in this market regime.

        Example:
          - Bull: 1.00x
          - Choppy: 0.50x
          - Bear: 0.00x (no longs)
        """
        policy = get_regime_policy(self.regime)
        return float(policy.risk_multiplier)


def _as_timestamp(x: Union[str, pd.Timestamp]) -> pd.Timestamp:
    """Convert a user input into a tz-naive pandas Timestamp."""
    ts = pd.Timestamp(x)
    # normalize timezone-aware timestamps to tz-naive
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts


def get_market_state(
    symbol: str = MARKET_SYMBOL,
    as_of_date: Optional[Union[str, pd.Timestamp]] = None,
) -> MarketState:
    """
    Load daily data for the given market symbol (default benchmark ticker),
    compute regime on the most recent bar (or on the bar at/just before as_of_date),
    and return a MarketState object.

    Rules:
      - If as_of_date is provided, we ONLY use rows with date <= as_of_date
        (no lookahead).
      - If no eligible rows, raise a ValueError (callers should catch and fail safe).
    """
    df = load_symbol_daily(symbol)

    # Basic sanity
    if df.empty:
        raise ValueError(f"No data loaded for market symbol: {symbol}")

    # Ensure date is Timestamp and sorted
    if "date" not in df.columns:
        raise ValueError(f"{symbol} dataframe missing required 'date' column")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    if as_of_date is not None:
        asof = _as_timestamp(as_of_date)
        df = df[df["date"] <= asof].copy()
        if df.empty:
            raise ValueError(f"No market rows at or before as_of_date={as_of_date!r}")

    # Indicators
    df["ma_fast"] = df["close"].rolling(REGIME_FAST_MA).mean()
    df["ma_slow"] = df["close"].rolling(REGIME_SLOW_MA).mean()

    # ATR(14) for context (not strictly required for regime label)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift(1)).abs()
    tr3 = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr"] = tr.rolling(REGIME_ATR_WINDOW).mean()

    latest = df.iloc[-1]
    close = float(latest["close"])
    ma_fast = float(latest["ma_fast"]) if pd.notna(latest["ma_fast"]) else np.nan
    ma_slow = float(latest["ma_slow"]) if pd.notna(latest["ma_slow"]) else np.nan
    atr = float(latest["atr"]) if pd.notna(latest["atr"]) else None

    # Require enough data for MA signals
    if not np.isfinite(ma_fast) or not np.isfinite(ma_slow):
        raise ValueError(
            f"Insufficient data to compute regime MAs for {symbol}: "
            f"need at least {REGIME_SLOW_MA} rows."
        )

    # Simple regime rules:
    # - Bull if close > MA50 and MA50 > MA200
    # - Bear if close < MA50 and MA50 < MA200
    # - Else choppy
    if close > ma_fast and ma_fast > ma_slow:
        regime = "BULL"
    elif close < ma_fast and ma_fast < ma_slow:
        regime = "BEAR"
    else:
        regime = "CHOPPY"

    return MarketState(
        symbol=symbol,
        as_of_date=pd.Timestamp(latest["date"]),
        regime=regime,
        close=close,
        ma_fast=ma_fast,
        ma_slow=ma_slow,
        atr=atr,
    )


if __name__ == "__main__":
    # Usage:
    #   python -u .\project_purple\market_state.py
    #   python -u .\project_purple\market_state.py 2023-12-29
    arg_as_of = sys.argv[1] if len(sys.argv) > 1 else None
    state = get_market_state(as_of_date=arg_as_of)

    if state.regime == "BULL":
        regime_str = f"{GREEN}{state.regime}{RESET}"
    elif state.regime == "BEAR":
        regime_str = f"{RED}{state.regime}{RESET}"
    else:
        regime_str = f"{YELLOW}{state.regime}{RESET}"

    trade_long_str = f"{GREEN}True{RESET}" if state.trade_long else f"{RED}False{RESET}"

    print("\n=== MARKET STATE ===")
    print(f"Symbol:         {state.symbol}")
    print(f"As of date:     {state.as_of_date}")

    print(f"\nRegime:         {regime_str}")
    print(f"Trade long:     {trade_long_str}")
    print(f"Risk mult:      {state.risk_multiplier:.2f}x")

    print(f"\nClose:          {state.close:.2f}")
    print(f"50 MA:          {state.ma_fast:.2f}")
    print(f"200 MA:         {state.ma_slow:.2f}")

    if state.atr is not None:
        print(f"ATR(14):        {state.atr:.2f}")
    else:
        print("ATR(14):        N/A")

    print()
