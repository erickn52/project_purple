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
    atr_multiple_stop: float = 2.0
    atr_multiple_target: float = 3.0

    # Maximum time in a trade (bars)
    max_holding_days: int = 10


strategy_settings = StrategySettings()
