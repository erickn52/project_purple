# project_purple/trade_plan.py

from __future__ import annotations

from typing import Any, Optional, Union

import csv
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Prefer package-relative imports, with a fallback for direct script execution.
try:
    from .market_state import get_market_state
    from .regime_risk import get_regime_policy
    from .scanner_simple import build_universe
    from .data_loader import load_symbol_daily
    from .backtest_v2 import SLIPPAGE_PCT, add_atr, add_momentum_pullback_signals
    from .risk import RiskConfig, calculate_risk_for_trade
except ImportError:  # pragma: no cover
    from market_state import get_market_state
    from regime_risk import get_regime_policy
    from scanner_simple import build_universe
    from data_loader import load_symbol_daily
    from backtest_v2 import SLIPPAGE_PCT, add_atr, add_momentum_pullback_signals
    from risk import RiskConfig, calculate_risk_for_trade


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
    Build a daily swing trade plan (long-only).

    - Does NOT print; returns structured data.
    - If return_meta=True: always returns a dict with status + details.
    - If return_meta=False: returns plan dict on success else None.
    """
    now_utc = datetime.now(timezone.utc).isoformat()

    # ---- 1) Market state (defensive) ----
    try:
        state = get_market_state(as_of_date=as_of_date)
    except Exception as e:
        meta: dict = {
            "status": "market_state_error",
            "timestamp_utc": now_utc,
            "as_of_input": as_of_date,
            "regime": "UNKNOWN",
            "trade_long_allowed": False,
            "risk_multiplier": 0.0,
            "policy_error": "",
            "symbol": "",
            "as_of_date": None,
            "risk_config": None,
            "entry_price": None,
            "stop_price": None,
            "target_price": None,
            "shares": None,
            "dollars_at_risk": None,
            "note": "market_state_error",
            "market_state_error": str(e),
        }
        _append_trade_journal_row(
            {
                "timestamp_utc": now_utc,
                "as_of_date": str(as_of_date) if as_of_date is not None else "",
                "regime": "UNKNOWN",
                "trade_long_allowed": False,
                "risk_multiplier": 0.0,
                "symbol": "",
                "entry_price": "",
                "stop_price": "",
                "target_price": "",
                "shares": "",
                "dollars_at_risk": "",
                "note": "market_state_error",
            }
        )
        return meta if return_meta else None

    # Centralized regime policy
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

    # ---- 2) Universe selection ----
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

    # ---- 3) Load symbol + indicators ----
    df = load_symbol_daily(symbol, as_of_date=as_of_date)
    df = add_atr(df)
    df = add_momentum_pullback_signals(df)

    usable = df.dropna(subset=["close", "atr"])
    if usable.empty:
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

    last = usable.iloc[-1]

    # ---- 3.5) ENTRY SIGNAL ENFORCEMENT ----
    # Only plan a trade when today's signal says "on".
    sig_val = last.get("signal") if isinstance(last, pd.Series) else None
    try:
        sig_int = int(sig_val) if sig_val is not None and not pd.isna(sig_val) else 0
    except Exception:
        sig_int = 0

    if sig_int != 1:
        meta["status"] = "signal_off"
        meta["note"] = "signal_off"
        meta["as_of_date"] = pd.to_datetime(last.get("date")) if "date" in last else None

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
                "note": "signal_off",
            }
        )
        return meta if return_meta else None

    # ---- 4) Risk config + sizing ----
    close = float(last["close"])
    atr = float(last["atr"])

    BASE_RISK_PER_TRADE_PCT = 0.0075  # 0.75% baseline (we'll centralize later)

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
    shares_raw = _risk_value(risk, "shares")
    dollars_at_risk = _risk_value(risk, "dollars_at_risk", fallback_key="dollar_risk")

    # ---- STEP 3: Block zero-share "plans" ----
    # Risk sizing can legitimately return 0 shares when even 1 share violates your risk rules.
    # In that case, the system must return NO TRADE (not "planned").
    try:
        shares_int = int(float(shares_raw)) if shares_raw is not None else 0
    except Exception:
        shares_int = 0

    if shares_int <= 0:
        meta.update(
            {
                "status": "zero_shares",
                "note": "zero_shares",
                "as_of_date": pd.to_datetime(last["date"]) if "date" in last else None,
                "risk_config": risk_config,
                "entry_price": float(entry_price) if entry_price is not None else None,
                "stop_price": float(stop_price) if stop_price is not None else None,
                "target_price": float(target_price) if target_price is not None else None,
                "shares": 0,
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
                "shares": 0,
                "dollars_at_risk": dollars_at_risk if dollars_at_risk is not None else "",
                "note": "zero_shares",
            }
        )
        return meta if return_meta else None

    meta.update(
        {
            "status": "planned",
            "note": "planned",
            "as_of_date": pd.to_datetime(last["date"]) if "date" in last else None,
            "risk_config": risk_config,
            "entry_price": float(entry_price) if entry_price is not None else None,
            "stop_price": float(stop_price) if stop_price is not None else None,
            "target_price": float(target_price) if target_price is not None else None,
            "shares": shares_int,
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
            "shares": shares_int,
            "dollars_at_risk": dollars_at_risk if dollars_at_risk is not None else "",
            "note": meta.get("note", ""),
        }
    )

    return meta if return_meta else {
        "symbol": symbol,
        "as_of_date": meta["as_of_date"],
        "regime": state.regime,
        "risk_multiplier": risk_mult,
        "risk_config": risk_config,
        "entry_price": meta["entry_price"],
        "stop_price": meta["stop_price"],
        "target_price": meta["target_price"],
        "shares": shares_int,
        "dollars_at_risk": dollars_at_risk,
    }
