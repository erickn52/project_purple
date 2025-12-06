# project_purple/settings.py

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StrategySettings:
    # Fraction of account equity risked per trade (in R terms)
    # 0.0075 = 0.75% of equity per trade
    risk_per_trade: float = 0.0075

    # Maximum *position size* as a fraction of equity
    # 0.20 = no single trade can exceed 20% of account value
    max_position_pct: float = 0.20

    # Stop/target expressed in ATR units
    # We’re increasing stop distance so NVDA-style volatility doesn’t
    # immediately knock us out, and targeting a larger reward.
    atr_multiple_stop: float = 3.0
    atr_multiple_target: float = 6.0

    # Trailing stop expressed in ATR units (used to ratchet up the stop)
    atr_trailing_multiple: float = 3.0

    # Volatility filter: compare fast ATR to a slower ATR average
    # Only trade when atr_14 <= atr_vol_max_mult * atr_slow
    atr_vol_window: int = 50
    atr_vol_max_mult: float = 1.5

    # Maximum time in a trade (bars) – safety net, not primary exit
    max_holding_days: int = 30

    # Use market-wide regime filter (SPY trend) if available
    use_market_regime: bool = True


strategy_settings = StrategySettings()
