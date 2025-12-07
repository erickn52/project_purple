from dataclasses import dataclass
from typing import Dict


@dataclass
class RiskConfig:
    """
    Configuration for risk management and position sizing.
    All values here are in terms of percentages or ATR multiples.

    You can tweak these defaults later.
    """
    # How much of the account you are willing to risk per trade (e.g. 0.01 = 1%)
    risk_per_trade_pct: float = 0.01

    # How many ATRs below entry the stop should be (for long trades)
    atr_stop_multiple: float = 1.5

    # How many ATRs above entry the initial target should be (for long trades)
    atr_target_multiple: float = 3.0

    # Maximum size of any single position as a % of total equity (e.g. 0.2 = 20%)
    max_position_pct_of_equity: float = 0.20

    # Minimum number of shares; helps avoid weird 0-share trades due to rounding
    min_shares: int = 1


def calculate_risk_for_trade(
    entry_price: float,
    atr: float,
    account_equity: float,
    config: RiskConfig | None = None
) -> Dict[str, float]:
    """
    Given an entry price, ATR, and account equity, compute:
      - stop price
      - target price
      - risk per share
      - dollar risk for the trade
      - position size in shares
      - position notional value

    This assumes a LONG trade. We can extend to shorts later.
    """

    if config is None:
        config = RiskConfig()

    if entry_price <= 0:
        raise ValueError("entry_price must be > 0")

    if atr <= 0:
        raise ValueError("ATR must be > 0 to compute volatility-based risk")

    if account_equity <= 0:
        raise ValueError("account_equity must be > 0")

    # 1. Stop and target based on ATR multiples
    stop_price = entry_price - config.atr_stop_multiple * atr
    target_price = entry_price + config.atr_target_multiple * atr

    # 2. Risk per share is distance from entry to stop
    risk_per_share = entry_price - stop_price
    if risk_per_share <= 0:
        raise ValueError(
            "Calculated risk_per_share is not positive. "
            "Check ATR and atr_stop_multiple values."
        )

    # 3. Dollar risk you are allowed for this trade
    allowed_dollar_risk = account_equity * config.risk_per_trade_pct

    # 4. Raw position size from allowed risk / risk per share
    raw_shares = int(allowed_dollar_risk // risk_per_share)

    # 5. Cap position size by max % of equity
    max_position_value = account_equity * config.max_position_pct_of_equity
    if entry_price > 0:
        max_position_shares = int(max_position_value // entry_price)
    else:
        max_position_shares = raw_shares

    # The final number of shares is the minimum of:
    # - what risk allows
    # - what max position size allows
    # but at least min_shares (unless risk is so small that even 1 share is too risky)
    shares = min(raw_shares, max_position_shares)

    if shares < config.min_shares:
        # If even 1 share breaks the risk rules, we signal that by returning 0 shares.
        shares = 0

    position_value = shares * entry_price
    dollar_risk = shares * risk_per_share

    return {
        "entry_price": entry_price,
        "atr": atr,
        "stop_price": stop_price,
        "target_price": target_price,
        "risk_per_share": risk_per_share,
        "allowed_dollar_risk": allowed_dollar_risk,
        "shares": float(shares),
        "position_value": position_value,
        "dollar_risk": dollar_risk,
    }


def calculate_R(
    exit_price: float,
    entry_price: float,
    stop_price: float,
) -> float:
    """
    Calculate the R-multiple for a completed trade.

    R = (exit_price - entry_price) / (entry_price - stop_price)

    Positive R = profit, negative R = loss.
    """
    risk_per_share = entry_price - stop_price
    if risk_per_share <= 0:
        raise ValueError("risk_per_share must be positive to compute R.")

    return (exit_price - entry_price) / risk_per_share
