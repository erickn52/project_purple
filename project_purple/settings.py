# project_purple/settings.py
from dataclasses import dataclass


@dataclass
class StrategySettings:
    # Universe filters
    min_price: float = 12.0
    max_price: float = 60.0

    # Average dollar volume filter (price * volume)
    # We want reasonably liquid names with tight spreads.
    min_avg_dollar_volume: float = 2_000_000  # 2 million USD

    # Sectors / industries we want to avoid completely
    exclude_sectors: tuple[str, ...] = ("Biotechnology", "Biotech")

    # Risk management
    risk_per_trade: float = 0.0075  # 0.75% of account equity per trade
    max_positions: int = 8
    max_portfolio_risk: float = 0.20  # 20% of equity at risk if all stops are hit

    # Stops & targets (use ATR for volatility-adjusted exits)
    atr_period: int = 14
    atr_multiple_stop: float = 2.0
    atr_multiple_target: float = 3.0

    # Holding & regime filters
    max_holding_days: int = 10           # cap on how long we hold a swing
    min_trend_lookback_days: int = 20    # for basic "is it trending?" checks


strategy_settings = StrategySettings()
