# project_purple/trade_plan.py

from __future__ import annotations

from typing import Any, Optional, Union

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


def _risk_value(risk: Any, key: str, fallback_key: str | None = None) -> Any:
    """Read risk values from either a dict return or an object-with-attributes return."""
    if isinstance(risk, dict):
        if key in risk:
            return risk[key]
        if fallback_key is not None and fallback_key in risk:
            return risk[fallback_key]
        return None
    if hasattr(risk, key):
        return getattr(risk, key)
    if fallback_key is not None and hasattr(risk, fallback_key):
        return getattr(risk, fallback_key)
    return None


def _fmt_price(x: Any) -> str:
    try:
        return f"{float(x):.2f}"
    except Exception:
        return ""


def _fmt_money(x: Any) -> str:
    try:
        return f"{float(x):,.2f}"
    except Exception:
        return ""


def _fmt_pct(x: Any) -> str:
    try:
        return f"{float(x) * 100:.3f}%"
    except Exception:
        return ""


def _fmt_date(x: Any) -> str:
    try:
        ts = pd.to_datetime(x)
        return ts.strftime("%Y-%m-%d")
    except Exception:
        return str(x)


def _print_trade_plan_rows(plan: dict) -> None:
    """Pretty-print the plan in row format (terminal-friendly)."""
    rc = plan.get("risk_config")

    print("\n=== TRADE PLAN (Policy A) ===")
    print(f"{'symbol':>18}: {plan.get('symbol', '')}")
    print(f"{'as_of_date':>18}: {_fmt_date(plan.get('as_of_date'))}")
    print(f"{'regime':>18}: {plan.get('regime', '')}")
    print(f"{'risk_multiplier':>18}: {plan.get('risk_multiplier', '')}")

    # Risk config printed as a small block
    if rc is not None:
        print(f"{'risk_config':>18}:")
        print(f"{'':>18}  risk_per_trade_pct      = {_fmt_pct(getattr(rc, 'risk_per_trade_pct', None))}")
        print(f"{'':>18}  atr_stop_multiple       = {getattr(rc, 'atr_stop_multiple', '')}")
        print(f"{'':>18}  atr_target_multiple     = {getattr(rc, 'atr_target_multiple', '')}")
        print(f"{'':>18}  max_pos_pct_of_equity   = {_fmt_pct(getattr(rc, 'max_position_pct_of_equity', None))}")
        print(f"{'':>18}  min_shares              = {getattr(rc, 'min_shares', '')}")

    print(f"{'entry_price':>18}: {_fmt_price(plan.get('entry_price'))}")
    print(f"{'stop_price':>18}: {_fmt_price(plan.get('stop_price'))}")
    print(f"{'target_price':>18}: {_fmt_price(plan.get('target_price'))}")

    shares = plan.get("shares")
    try:
        shares_disp = str(int(float(shares))) if shares is not None else ""
    except Exception:
        shares_disp = str(shares) if shares is not None else ""

    print(f"{'shares':>18}: {shares_disp}")
    print(f"{'dollars_at_risk':>18}: {_fmt_money(plan.get('dollars_at_risk'))}")


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
    # (HR-3 already confirmed complete in Codex report.)
    df = load_symbol_daily(symbol, as_of_date=as_of_date)
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

    entry_price = float(_risk_value(risk, "entry_price") or 0.0)
    stop_price = float(_risk_value(risk, "stop_price") or 0.0)
    target_price = float(_risk_value(risk, "target_price") or 0.0)
    shares = _risk_value(risk, "shares")
    dollars_at_risk = _risk_value(risk, "dollars_at_risk", fallback_key="dollar_risk")

    plan = {
        "symbol": symbol,
        "as_of_date": pd.to_datetime(last["date"]),
        "regime": state.regime,
        "risk_multiplier": risk_mult,
        "risk_config": risk_config,
        "entry_price": entry_price,
        "stop_price": stop_price,
        "target_price": target_price,
        "shares": shares,
        "dollars_at_risk": dollars_at_risk,
    }

    _append_trade_journal_row(
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "as_of_date": str(as_of_date),
            "regime": state.regime,
            "trade_long_allowed": trade_long_allowed,
            "risk_multiplier": risk_mult,
            "symbol": symbol,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "target_price": target_price,
            "shares": shares,
            "dollars_at_risk": dollars_at_risk,
            "note": "planned",
        }
    )

    _print_trade_plan_rows(plan)
    return plan
