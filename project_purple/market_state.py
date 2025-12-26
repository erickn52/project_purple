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


# Basic parameters for regime detection
REGIME_FAST_MA = 50
REGIME_SLOW_MA = 200
REGIME_ATR_WINDOW = 14

MARKET_SYMBOL = "SPY"  # we can parameterize later if needed

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
        Allowed in BULL or CHOPPY, blocked in BEAR.

        Delegates to regime_risk.py (single source of truth).
        """
        return bool(get_regime_policy(self.regime).trade_long)

    @property
    def market_long_ok(self) -> bool:
        """Backwards-compatible alias used by other modules."""
        return self.trade_long

    @property
    def risk_multiplier(self) -> float:
        """
        Multiplier applied to base risk per trade based on market regime.

        Delegates to regime_risk.py (single source of truth).
        """
        return float(get_regime_policy(self.regime).risk_multiplier)


def _compute_atr(df: pd.DataFrame, window: int) -> pd.Series:
    """Classic True Range ATR, simple moving average."""
    prev_close = df["close"].shift(1)

    high_low = df["high"] - df["low"]
    high_prev = (df["high"] - prev_close).abs()
    low_prev = (df["low"] - prev_close).abs()

    tr = pd.concat([high_low, high_prev, low_prev], axis=1).max(axis=1)
    atr = tr.rolling(window=window, min_periods=window).mean()
    return atr


def _normalize_as_of_date(as_of_date: Optional[Union[str, pd.Timestamp]]) -> Optional[pd.Timestamp]:
    if as_of_date is None:
        return None
    ts = pd.to_datetime(as_of_date, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid as_of_date: {as_of_date!r}")
    return ts


def get_market_state(
    symbol: str = MARKET_SYMBOL,
    as_of_date: Optional[Union[str, pd.Timestamp]] = None,
) -> MarketState:
    """
    Load daily data for the given market symbol (default SPY),
    compute regime on the most recent bar (or on the bar at/just before as_of_date),
    and return a MarketState object.

    HR2:
      - If as_of_date is provided, we ONLY use rows with date <= as_of_date.
        This prevents lookahead bias in historical simulation.

    NOTE: load_symbol_daily() is responsible for applying the inclusive cutoff.
    """

    df = load_symbol_daily(symbol, as_of_date=as_of_date)
    if df.empty:
        raise ValueError(f"No data loaded for symbol {symbol}")

    df = df.copy()

    # Normalize / sanitize dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "open", "high", "low", "close"]).copy()
    df = df.sort_values("date").reset_index(drop=True)

    # Data loader already applies as_of_date cutoff (inclusive) when provided.

    if df.empty:
        raise ValueError(f"No usable rows for symbol {symbol} as of {as_of_date!r}")

    # Indicators
    df["ma_fast"] = df["close"].rolling(window=REGIME_FAST_MA, min_periods=REGIME_FAST_MA).mean()
    df["ma_slow"] = df["close"].rolling(window=REGIME_SLOW_MA, min_periods=REGIME_SLOW_MA).mean()
    df["atr"] = _compute_atr(df, REGIME_ATR_WINDOW)

    last = df.iloc[-1]

    close = float(last["close"])
    ma_fast = float(last["ma_fast"]) if not np.isnan(last["ma_fast"]) else float("nan")
    ma_slow = float(last["ma_slow"]) if not np.isnan(last["ma_slow"]) else float("nan")
    atr = float(last["atr"]) if not np.isnan(last["atr"]) else None
    bar_date = pd.to_datetime(last["date"])

    # Regime requires slow MA to exist; otherwise results are undefined.
    if np.isnan(ma_slow):
        raise ValueError(
            f"Insufficient history to compute {REGIME_SLOW_MA}MA for {symbol}"
            + (f" as of {as_of_date!r}" if as_of_date is not None else "")
        )

    # --- Simple regime classification ----------------------------------------
    if close > ma_slow and (not np.isnan(ma_fast)) and ma_fast > ma_slow:
        regime = "BULL"
    elif close < ma_slow:
        regime = "BEAR"
    else:
        regime = "CHOPPY"

    return MarketState(
        symbol=symbol,
        as_of_date=bar_date,
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
