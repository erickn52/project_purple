# project_purple/trade_plan.py

from __future__ import annotations

from typing import Optional, Union

import csv
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# IMPORTANT:
# This module is often imported as: from project_purple.trade_plan import build_trade_plan
# When imported that way, absolute imports like "from scanner_simple import ..." fail
# because scanner_simple is inside the project_purple package.
# So we prefer package-relative imports, with a fallback for running from within the
# package directory (or with sys.path tweaks).
try:
    from .scanner_simple import build_universe
    from .market_state import get_market_state
    from .data_loader import load_symbol_daily
    from .risk import RiskConfig, calculate_risk_for_trade
    from .backtest_v2 import SLIPPAGE_PCT, add_atr, add_momentum_pullback_signals
except ImportError:  # pragma: no cover
    from scanner_simple import build_universe
    from market_state import get_market_state
    from data_loader import load_symbol_daily
    from risk import RiskConfig, calculate_risk_for_trade
    from backtest_v2 import SLIPPAGE_PCT, add_atr, add_momentum_pullback_signals

# Centralized regime policy (single source of truth).
try:
    from .regime_risk import get_regime_policy
except ImportError:  # pragma: no cover
    try:
        from regime_risk import get_regime_policy
    except ModuleNotFoundError:  # pragma: no cover
        from project_purple.regime_risk import get_regime_policy


BASE_RISK_PER_TRADE_PCT = 0.01  # will be scaled by regime risk multiplier


def _ensure_logs_dir() -> Path:
    """Ensure ./logs exists and return the Path (repo root)."""
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


def _append_trade_journal_row(row: dict) -> None:
    """Append a single journal row to ./logs/trade_journal.csv (create if missing)."""
    journal_path = _ensure_logs_dir() / "trade_journal.csv"

    fieldnames = [
        "timestamp_utc",
        "as_of_date",
        "regime",
        "trade_long_allowed",
        "risk_multiplier",
        "symbol",
        "entry_price",
        "stop_price",
        "target_price",
        "shares",
        "dollars_at_risk",
        "note",
    ]

    file_exists = journal_path.exists()
    with journal_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def build_trade_plan(as_of_date: Optional[Union[str, pd.Timestamp]] = None) -> Optional[dict]:
    """
    Build a daily swing trade plan (Policy A: 1 position at a time; long-only).

    Notes:
      - Risk per trade is BASE_RISK_PER_TRADE_PCT scaled by regime_risk multiplier.
      - as_of_date is forwarded to regime + scanner to avoid lookahead bias.
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
        risk_mult = 0.0
        trade_long_allowed = False

    # Simple ANSI colors (no dependency)
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"

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
        _append_trade_journal_row(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "as_of_date": str(as_of_date),
                "regime": state.regime,
                "trade_long_allowed": trade_long_allowed,
                "risk_multiplier": risk_mult,
                "symbol": "",
                "entry_price": "",
                "stop_price": "",
                "target_price": "",
                "shares": "",
                "dollars_at_risk": "",
                "note": "blocked_by_regime",
            }
        )
        return None

    universe = build_universe(as_of_date=as_of_date)
    if not universe:
        print("\nNo symbols in universe. No trade plan.")
        _append_trade_journal_row(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "as_of_date": str(as_of_date),
                "regime": state.regime,
                "trade_long_allowed": trade_long_allowed,
                "risk_multiplier": risk_mult,
                "symbol": "",
                "entry_price": "",
                "stop_price": "",
                "target_price": "",
                "shares": "",
                "dollars_at_risk": "",
                "note": "empty_universe",
            }
        )
        return None

    # Policy A currently takes the top symbol.
    symbol = universe[0]

    # NOTE:
    # We intentionally rely on data_loader.load_symbol_daily() for validation.
    df = load_symbol_daily(symbol)
    df = df.sort_values("date").reset_index(drop=True)

    # Indicators + signals reused from backtest module for consistency.
    df = add_atr(df, period=14)
    df = add_momentum_pullback_signals(df, ma_fast=20, ma_slow=50)
    df = df.dropna(subset=["atr", "ma_fast", "ma_slow"]).reset_index(drop=True)

    if df.empty:
        print("\nNo usable rows after indicators. No trade plan.")
        _append_trade_journal_row(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "as_of_date": str(as_of_date),
                "regime": state.regime,
                "trade_long_allowed": trade_long_allowed,
                "risk_multiplier": risk_mult,
                "symbol": symbol,
                "entry_price": "",
                "stop_price": "",
                "target_price": "",
                "shares": "",
                "dollars_at_risk": "",
                "note": "no_usable_rows_after_indicators",
            }
        )
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
        equity=100_000.0,  # planning equity placeholder; execution layer will replace
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

    _append_trade_journal_row(
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "as_of_date": str(as_of_date),
            "regime": state.regime,
            "trade_long_allowed": trade_long_allowed,
            "risk_multiplier": risk_mult,
            "symbol": symbol,
            "entry_price": risk.entry_price,
            "stop_price": risk.stop_price,
            "target_price": risk.target_price,
            "shares": risk.shares,
            "dollars_at_risk": risk.dollars_at_risk,
            "note": "planned",
        }
    )

    print("\n=== TRADE PLAN (Policy A) ===")
    print(plan)

    return plan
