from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from data_loader import load_symbol_daily

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
    regime: str          # "BULL", "BEAR", "CHOPPY"
    close: float
    ma_fast: float
    ma_slow: float
    atr: Optional[float]

    @property
    def trade_long(self) -> bool:
        """
        True when it is OK to take new long trades.
        For now: allowed in BULL or CHOPPY, blocked in BEAR.
        """
        return self.regime != "BEAR"

    @property
    def market_long_ok(self) -> bool:
        """
        Backwards-compatible alias used by other modules (signals, etc.).
        Same meaning as trade_long.
        """
        return self.trade_long


def _compute_atr(df: pd.DataFrame, window: int) -> pd.Series:
    """Classic True Range ATR, simple moving average."""
    prev_close = df["close"].shift(1)

    high_low = df["high"] - df["low"]
    high_prev = (df["high"] - prev_close).abs()
    low_prev = (df["low"] - prev_close).abs()

    tr = pd.concat([high_low, high_prev, low_prev], axis=1).max(axis=1)
    atr = tr.rolling(window=window, min_periods=window).mean()
    return atr


def get_market_state(symbol: str = MARKET_SYMBOL) -> MarketState:
    """
    Load daily data for the given market symbol (default SPY),
    compute regime on the most recent bar, and return a MarketState object.
    """
    df = load_symbol_daily(symbol)

    if df.empty:
        raise ValueError(f"No data loaded for symbol {symbol}")

    df = df.copy()
    df["ma_fast"] = df["close"].rolling(
        window=REGIME_FAST_MA, min_periods=REGIME_FAST_MA
    ).mean()
    df["ma_slow"] = df["close"].rolling(
        window=REGIME_SLOW_MA, min_periods=REGIME_SLOW_MA
    ).mean()
    df["atr"] = _compute_atr(df, REGIME_ATR_WINDOW)

    last = df.iloc[-1]

    close = float(last["close"])
    ma_fast = float(last["ma_fast"])
    ma_slow = float(last["ma_slow"])
    atr = float(last["atr"]) if not np.isnan(last["atr"]) else None
    as_of_date = pd.to_datetime(last["date"])

    # --- Simple regime classification ----------------------------------------
    if not np.isnan(ma_slow) and close > ma_slow and ma_fast > ma_slow:
        regime = "BULL"
    elif not np.isnan(ma_slow) and close < ma_slow:
        regime = "BEAR"
    else:
        regime = "CHOPPY"

    return MarketState(
        symbol=symbol,
        as_of_date=as_of_date,
        regime=regime,
        close=close,
        ma_fast=ma_fast,
        ma_slow=ma_slow,
        atr=atr,
    )


if __name__ == "__main__":
    state = get_market_state()

    # Color the regime
    if state.regime == "BULL":
        regime_str = f"{GREEN}{state.regime}{RESET}"
    elif state.regime == "BEAR":
        regime_str = f"{RED}{state.regime}{RESET}"
    else:
        regime_str = f"{YELLOW}{state.regime}{RESET}"

    trade_long_str = (
        f"{GREEN}True{RESET}" if state.trade_long else f"{RED}False{RESET}"
    )

    print("\n=== MARKET STATE ===")
    print(f"Symbol:         {state.symbol}")
    print(f"As of date:     {state.as_of_date}")

    print(f"\nRegime:         {regime_str}")
    print(f"Trade long:     {trade_long_str}")

    print(f"\nClose:          {state.close:.2f}")
    print(f"50 MA:          {state.ma_fast:.2f}")
    print(f"200 MA:         {state.ma_slow:.2f}")

    if state.atr is not None:
        print(f"ATR(14):        {state.atr:.2f}")
    else:
        print("ATR(14):        N/A")

    print()
