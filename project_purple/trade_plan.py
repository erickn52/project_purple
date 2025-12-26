# project_purple/trade_plan.py

from __future__ import annotations

from typing import Optional, Union

import pandas as pd

from scanner_simple import build_universe
from market_state import get_market_state
from data_loader import load_symbol_daily
from risk import RiskConfig, calculate_risk_for_trade
from backtest_v2 import SLIPPAGE_PCT, add_atr, add_momentum_pullback_signals

# Centralized regime policy (single source of truth).
# Support running both:
# - from inside project_purple/: python trade_plan.py (via sys.path or package context)
# - from repo root with sys.path tweaks / module import
try:
    from regime_risk import get_regime_policy
except ModuleNotFoundError:  # pragma: no cover
    from project_purple.regime_risk import get_regime_policy


BASE_RISK_PER_TRADE_PCT = 0.01  # will be scaled by regime risk multiplier


GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def build_trade_plan(
    as_of_date: Optional[Union[str, pd.Timestamp]] = None,
):
    """
    Build a single-symbol trade plan (Policy A):
      - Determine market regime from SPY (via market_state)
      - If trade_long is blocked, return None
      - Build a trading universe (scanner_simple)
      - Pick the top-ranked symbol (first in list)
      - Compute ATR + signals and return a dictionary plan

    Notes:
      - Risk per trade is BASE_RISK_PER_TRADE_PCT scaled by regime_risk multiplier.
      - as_of_date is forwarded to regime + scanner + symbol data loads to avoid lookahead bias.
    """
    state = get_market_state(as_of_date=as_of_date)

    # Centralized regime rules (single source of truth)
    try:
        policy = get_regime_policy(state.regime)
        risk_mult = float(policy.risk_multiplier)
        trade_long_allowed = bool(policy.trade_long)
    except Exception as e:
        # Defensive fallback (should rarely happen)
        print("\nWARNING: Could not resolve regime policy from regime_risk.py.")
        print(f"Regime: {state.regime!r}")
        print(f"Reason: {e}")
        risk_mult = 0.0 if state.regime == "BEAR" else 1.0
        trade_long_allowed = (state.regime != "BEAR")

    if state.regime == "BULL":
        regime_str = f"{GREEN}{state.regime}{RESET}"
    elif state.regime == "BEAR":
        regime_str = f"{RED}{state.regime}{RESET}"
    else:
        regime_str = f"{YELLOW}{state.regime}{RESET}"

    trade_long_str = f"{GREEN}True{RESET}" if trade_long_allowed else f"{RED}False{RESET}"

    print("\n=== MARKET STATE ===")
    print(f"Regime:     {regime_str}")
    print(f"Trade long: {trade_long_str}")
    print(f"Risk mult:  {risk_mult:.2f}x")

    if not trade_long_allowed:
        print("\nBlocked: trade_long=False under current regime policy. No trade plan.")
        return None

    universe = build_universe(as_of_date=as_of_date)
    if not universe:
        print("\nNo symbols in universe. No trade plan.")
        return None

    symbol = universe[0]
    df = load_symbol_daily(symbol)
    df = df.sort_values("date").reset_index(drop=True)

    df = add_atr(df, period=14)
    df = add_momentum_pullback_signals(df, ma_fast=20, ma_slow=50)
    df = df.dropna(subset=["atr", "ma_fast", "ma_slow"]).reset_index(drop=True)

    if df.empty:
        print("\nNo usable rows after indicators. No trade plan.")
        return None

    last = df.iloc[-1]
    close = float(last["close"])
    atr = float(last["atr"])

    risk_config = RiskConfig(
        risk_per_trade_pct=BASE_RISK_PER_TRADE_PCT * risk_mult,
        atr_stop_multiple=1.5,
        atr_target_multiple=3.0,
        max_position_pct_of_equity=0.20,
        min_shares=1,
    )

    # Compute position sizing & brackets
    risk = calculate_risk_for_trade(
        entry_price=close * (1.0 + SLIPPAGE_PCT),
        atr=atr,
        risk_config=risk_config,
        equity=100_000.0,  # placeholder planning equity; execution layer will replace
    )

    plan = {
        "symbol": symbol,
        "as_of_date": pd.to_datetime(last["date"]),
        "regime": state.regime,
        "risk_multiplier": risk_mult,
        "risk_config": risk_config,
        "entry_price": risk.entry_price,
        "stop_price": risk.stop_price,
        "target_price": risk.target_price,
        "shares": risk.shares,
        "dollars_at_risk": risk.dollars_at_risk,
    }

    print("\n=== TRADE PLAN (Policy A) ===")
    print(plan)

    return plan
