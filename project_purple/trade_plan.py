# project_purple/trade_plan.py

from __future__ import annotations

import pandas as pd

from scanner_simple import build_universe
from market_state import get_market_state
from data_loader import load_symbol_daily
from risk import RiskConfig, calculate_risk_for_trade
from backtest_v2 import SLIPPAGE_PCT, add_atr, add_momentum_pullback_signals


def _regime_risk_mult(regime: str) -> float:
    if regime == "BULL":
        return 1.0
    if regime == "CHOPPY":
        return 0.5
    if regime == "BEAR":
        return 0.0
    return 1.0


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def main() -> None:
    # ------------------------------------------------------------
    # 1) Base settings (keep in sync with main.py defaults)
    # ------------------------------------------------------------
    account_equity = 100_000.0

    base_risk_config = RiskConfig(
        risk_per_trade_pct=0.01,         # base 1% (will be multiplied by regime)
        atr_stop_multiple=1.5,
        atr_target_multiple=3.0,
        max_position_pct_of_equity=0.20,
        min_shares=1,
    )

    # ------------------------------------------------------------
    # 2) Market regime (SPY) and risk multiplier
    # ------------------------------------------------------------
    print("\n===== TRADE PLAN =====")
    try:
        market_state = get_market_state("SPY")
        mult = _regime_risk_mult(market_state.regime)
        print("\n===== MARKET REGIME (SPY) =====")
        print(f"As of date : {market_state.as_of_date}")
        print(f"Regime     : {market_state.regime}")
        print(f"Trade long : {market_state.trade_long}")
        print(f"Close      : {market_state.close:.2f}")
        print(f"50 MA      : {market_state.ma_fast:.2f}")
        print(f"200 MA     : {market_state.ma_slow:.2f}")
        if market_state.atr is not None:
            print(f"ATR(14)    : {market_state.atr:.2f}")
        else:
            print("ATR(14)    : N/A")
    except Exception as e:
        market_state = None
        mult = 1.0
        print("\nWARNING: Could not determine market regime from SPY.")
        print(f"Reason: {e}")

    # Apply multiplier to a copy of the risk config
    risk_config = RiskConfig(
        risk_per_trade_pct=base_risk_config.risk_per_trade_pct * mult,
        atr_stop_multiple=base_risk_config.atr_stop_multiple,
        atr_target_multiple=base_risk_config.atr_target_multiple,
        max_position_pct_of_equity=base_risk_config.max_position_pct_of_equity,
        min_shares=base_risk_config.min_shares,
    )

    print(f"\nRisk mult  : {mult:.2f}x")
    print(f"Risk/trade : {risk_config.risk_per_trade_pct:.3%} (base {base_risk_config.risk_per_trade_pct:.3%})")

    # If BEAR, scanner should already give empty, but we hard-stop anyway
    if mult == 0.0:
        print("\nBEAR regime => no long trades. Trade plan ends.")
        return

    # ------------------------------------------------------------
    # 3) Build universe (this prints the scanner tables)
    # ------------------------------------------------------------
    print("\n===== BUILDING UNIVERSE (scanner_simple) =====")
    universe = build_universe()

    if not universe:
        print("\nUniverse empty. Nothing to plan.")
        return

    print(f"\nUniverse ({len(universe)}): {universe}")

    # ------------------------------------------------------------
    # 4) For each symbol: compute today's signal + detect new entry
    # ------------------------------------------------------------
    rows = []
    actionable_rows = []

    for rank, sym in enumerate(universe, start=1):
        try:
            df = load_symbol_daily(sym)
            if df.empty or len(df) < 60:
                rows.append({
                    "rank": rank,
                    "symbol": sym,
                    "status": "skip:insufficient_data",
                })
                continue

            # backtest_v2 expects columns: date/open/high/low/close/volume
            df = df.copy()
            df = df.sort_values("date").reset_index(drop=True)

            # Add ATR + MA signal used by backtest_v2 (momentum/pullback signal)
            df = add_atr(df, period=14)
            df = add_momentum_pullback_signals(df, ma_fast=20, ma_slow=50)

            # Need at least two rows to detect flip
            if len(df) < 2:
                rows.append({"rank": rank, "symbol": sym, "status": "skip:too_short"})
                continue

            last = df.iloc[-1]
            prev = df.iloc[-2]

            last_close = _safe_float(last["close"])
            last_atr = _safe_float(last.get("atr", float("nan")))
            ma_fast = _safe_float(last.get("ma_fast", float("nan")))
            ma_slow = _safe_float(last.get("ma_slow", float("nan")))

            prev_signal = int(prev.get("signal", 0))
            signal = int(last.get("signal", 0))

            actionable = (prev_signal == 0 and signal == 1)

            # Summary row for the whole universe
            rows.append({
                "rank": rank,
                "symbol": sym,
                "last_close": last_close,
                "atr": last_atr,
                "ma_fast": ma_fast,
                "ma_slow": ma_slow,
                "prev_signal": prev_signal,
                "signal": signal,
                "actionable_today": actionable,
                "status": "ok",
            })

            # If actionable, compute bracket plan (entry/stop/target/shares)
            if actionable:
                if not pd.notna(last_atr) or last_atr <= 0 or last_close <= 0:
                    actionable_rows.append({
                        "rank": rank,
                        "symbol": sym,
                        "status": "actionable_but_missing_atr_or_price",
                    })
                    continue

                # Match backtest_v2: entry is close plus slippage
                entry_price = last_close * (1.0 + SLIPPAGE_PCT)

                risk_info = calculate_risk_for_trade(
                    entry_price=entry_price,
                    atr=last_atr,
                    account_equity=account_equity,
                    config=risk_config,
                )

                shares = int(risk_info["shares"])
                if shares <= 0:
                    actionable_rows.append({
                        "rank": rank,
                        "symbol": sym,
                        "status": "actionable_but_shares=0 (risk too small or capped)",
                    })
                    continue

                actionable_rows.append({
                    "rank": rank,
                    "symbol": sym,
                    "entry_price": float(risk_info["entry_price"]),
                    "stop_price": float(risk_info["stop_price"]),
                    "target_price": float(risk_info["target_price"]),
                    "shares": shares,
                    "position_value": float(risk_info["position_value"]),
                    "allowed_dollar_risk": float(risk_info["allowed_dollar_risk"]),
                    "dollar_risk": float(risk_info["dollar_risk"]),
                    "atr": last_atr,
                    "status": "OK",
                })

        except Exception as e:
            rows.append({
                "rank": rank,
                "symbol": sym,
                "status": f"error:{e}",
            })

    # ------------------------------------------------------------
    # 5) Print results clearly
    # ------------------------------------------------------------
    df_u = pd.DataFrame(rows)

    print("\n===== UNIVERSE SIGNAL STATUS (today) =====")
    # Make it readable even if some rows are missing numeric cols
    cols = [c for c in [
        "rank", "symbol", "last_close", "atr", "ma_fast", "ma_slow",
        "prev_signal", "signal", "actionable_today", "status"
    ] if c in df_u.columns]
    df_u_print = df_u[cols].copy()

    for c in ["last_close", "atr", "ma_fast", "ma_slow"]:
        if c in df_u_print.columns:
            df_u_print[c] = pd.to_numeric(df_u_print[c], errors="coerce").round(2)

    print(df_u_print.to_string(index=False))

    df_a = pd.DataFrame(actionable_rows)

    print("\n===== ACTIONABLE ENTRIES TODAY (signal flipped 0 → 1) =====")
    if df_a.empty:
        print("None today. (That’s normal—entries won’t happen every day.)")
        return

    cols2 = [c for c in [
        "rank", "symbol", "entry_price", "stop_price", "target_price",
        "shares", "position_value", "allowed_dollar_risk", "dollar_risk", "atr", "status"
    ] if c in df_a.columns]
    df_a_print = df_a[cols2].copy()

    for c in ["entry_price", "stop_price", "target_price", "position_value", "allowed_dollar_risk", "dollar_risk", "atr"]:
        if c in df_a_print.columns:
            df_a_print[c] = pd.to_numeric(df_a_print[c], errors="coerce").round(2)

    print(df_a_print.sort_values(["rank", "symbol"]).to_string(index=False))


if __name__ == "__main__":
    main()
