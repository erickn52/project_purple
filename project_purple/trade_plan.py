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
    """Ensure ./logs exists and return the Path."""
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


def build_trade_plan(
    as_of_date: Optional[Union[str, pd.Timestamp]] = None,
    *,
    return_meta: bool = False,
) -> Optional[dict]:
    """
    Build a daily swing trade plan (Policy A: 1 position at a time; long-only).

    IMPORTANT:
    - This function does NOT print. It returns data only.
    - If return_meta=False (default): returns plan dict on success else None.
    - If return_meta=True: returns a dict ALWAYS (status + fields), even if blocked.

    Notes:
      - Risk per trade is BASE_RISK_PER_TRADE_PCT scaled by regime_risk multiplier.
      - as_of_date is forwarded to regime + scanner to avoid lookahead bias.
    """
    now_utc = datetime.now(timezone.utc).isoformat()

    # ---- 1) Market state ----
    state = get_market_state(as_of_date=as_of_date)

    # Centralized regime rules (single source of truth)
    try:
        policy = get_regime_policy(state.regime)
        risk_mult = float(policy.risk_multiplier)
        trade_long_allowed = bool(policy.trade_long)
        policy_error = ""
    except Exception as e:
        risk_mult = 0.0
        trade_long_allowed = False
        policy_error = str(e)

    meta: dict = {
        "status": "unknown",
        "timestamp_utc": now_utc,
        "as_of_input": as_of_date,
        "regime": state.regime,
        "trade_long_allowed": trade_long_allowed,
        "risk_multiplier": risk_mult,
        "policy_error": policy_error,
        # Plan fields (default empty)
        "symbol": "",
        "as_of_date": None,
        "risk_config": None,
        "entry_price": None,
        "stop_price": None,
        "target_price": None,
        "shares": None,
        "dollars_at_risk": None,
        "note": "",
    }

    # If policy resolution failed, treat it as blocked (defensive)
    if policy_error:
        meta["status"] = "blocked_policy_error"
        meta["note"] = "blocked_policy_error"
        _append_trade_journal_row(
            {
                "timestamp_utc": now_utc,
                "as_of_date": str(as_of_date) if as_of_date is not None else "",
                "regime": state.regime,
                "trade_long_allowed": trade_long_allowed,
                "risk_multiplier": risk_mult,
                "symbol": "",
                "entry_price": "",
                "stop_price": "",
                "target_price": "",
                "shares": "",
                "dollars_at_risk": "",
                "note": "blocked_policy_error",
            }
        )
        return meta if return_meta else None

    if not trade_long_allowed:
        meta["status"] = "blocked_by_regime"
        meta["note"] = "blocked_by_regime"
        _append_trade_journal_row(
            {
                "timestamp_utc": now_utc,
                "as_of_date": str(as_of_date) if as_of_date is not None else "",
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
        return meta if return_meta else None

    # ---- 2) Universe / pick top symbol ----
    universe = build_universe(as_of_date=as_of_date)
    if not universe:
        meta["status"] = "empty_universe"
        meta["note"] = "empty_universe"
        _append_trade_journal_row(
            {
                "timestamp_utc": now_utc,
                "as_of_date": str(as_of_date) if as_of_date is not None else "",
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
        return meta if return_meta else None

    symbol = universe[0]
    meta["symbol"] = symbol

    # ---- 3) Load data & compute indicators ----
    df = load_symbol_daily(symbol, as_of_date=as_of_date)
    df = df.sort_values("date").reset_index(drop=True)

    df = add_atr(df, period=14)
    df = add_momentum_pullback_signals(df, ma_fast=20, ma_slow=50)
    df = df.dropna(subset=["atr", "ma_fast", "ma_slow"]).reset_index(drop=True)

    if df.empty:
        meta["status"] = "no_usable_rows_after_indicators"
        meta["note"] = "no_usable_rows_after_indicators"
        _append_trade_journal_row(
            {
                "timestamp_utc": now_utc,
                "as_of_date": str(as_of_date) if as_of_date is not None else "",
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
        return meta if return_meta else None

    last = df.iloc[-1]
    close = float(last["close"])
    atr = float(last["atr"])

    # ---- 4) Risk config + sizing ----
    risk_config = RiskConfig(
        risk_per_trade_pct=BASE_RISK_PER_TRADE_PCT * risk_mult,
        atr_stop_multiple=1.5,
        atr_target_multiple=3.0,
        max_position_pct_of_equity=0.20,
        min_shares=1,
    )

    risk = calculate_risk_for_trade(
        entry_price=close * (1.0 + SLIPPAGE_PCT),
        atr=atr,
        risk_config=risk_config,
        equity=100_000.0,  # planning equity placeholder; execution layer will replace
    )

    entry_price = _risk_value(risk, "entry_price")
    stop_price = _risk_value(risk, "stop_price")
    target_price = _risk_value(risk, "target_price")
    shares = _risk_value(risk, "shares")
    dollars_at_risk = _risk_value(risk, "dollars_at_risk", fallback_key="dollar_risk")

    meta.update(
        {
            "status": "planned",
            "note": "planned",
            "as_of_date": pd.to_datetime(last["date"]),
            "risk_config": risk_config,
            "entry_price": float(entry_price) if entry_price is not None else None,
            "stop_price": float(stop_price) if stop_price is not None else None,
            "target_price": float(target_price) if target_price is not None else None,
            "shares": shares,
            "dollars_at_risk": dollars_at_risk,
        }
    )

    _append_trade_journal_row(
        {
            "timestamp_utc": now_utc,
            "as_of_date": str(as_of_date) if as_of_date is not None else "",
            "regime": state.regime,
            "trade_long_allowed": trade_long_allowed,
            "risk_multiplier": risk_mult,
            "symbol": symbol,
            "entry_price": meta["entry_price"] if meta["entry_price"] is not None else "",
            "stop_price": meta["stop_price"] if meta["stop_price"] is not None else "",
            "target_price": meta["target_price"] if meta["target_price"] is not None else "",
            "shares": shares if shares is not None else "",
            "dollars_at_risk": dollars_at_risk if dollars_at_risk is not None else "",
            "note": "planned",
        }
    )

    # Backward-compatible behavior:
    return meta if return_meta else {
        "symbol": symbol,
        "as_of_date": meta["as_of_date"],
        "regime": state.regime,
        "risk_multiplier": risk_mult,
        "risk_config": risk_config,
        "entry_price": meta["entry_price"],
        "stop_price": meta["stop_price"],
        "target_price": meta["target_price"],
        "shares": shares,
        "dollars_at_risk": dollars_at_risk,
    }
