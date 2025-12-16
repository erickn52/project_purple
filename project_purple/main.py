# project_purple/main.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd

from backtest_v2 import analyze_combined_trades, run_backtest_for_symbol
from market_state import get_market_state
from risk import RiskConfig


def _safe_import_build_universe():
    """
    Optional: if scanner_simple.build_universe() exists, we'll use it.
    If not, we'll fall back to a hard-coded list.
    """
    try:
        from scanner_simple import build_universe  # type: ignore
        return build_universe
    except Exception:
        return None


@dataclass
class PolicyAResult:
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    total_R: float
    avg_R: float
    final_equity: float
    max_drawdown_pct: float


def apply_portfolio_policy_A(
    combined_trades: pd.DataFrame,
    initial_equity: float,
    risk_per_trade_pct: float,
) -> PolicyAResult:
    """
    POLICY A (one position at a time):
      - On each entry_date, pick the trade with the highest entry_score (if available).
      - Only take a new trade after the previous trade has exited.
      - Update equity using R-multiple and fixed risk_per_trade_pct.

    Assumptions:
      - combined_trades includes: entry_date, exit_date, R
      - entry_score is optional (fallback: first trade of that day)
      - risk_per_trade_pct is a DECIMAL fraction (0.01 = 1%)
    """
    if combined_trades.empty:
        empty_curve = pd.DataFrame({"date": [], "equity": []})
        return PolicyAResult(
            trades=combined_trades.copy(),
            equity_curve=empty_curve,
            total_R=0.0,
            avg_R=0.0,
            final_equity=initial_equity,
            max_drawdown_pct=0.0,
        )

    df = combined_trades.copy()
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])

    # Pick best-per-day by entry_score if present; else keep all and we'll pick first.
    if "entry_score" in df.columns:
        day_best = (
            df.sort_values(["entry_date", "entry_score"], ascending=[True, False])
              .groupby("entry_date", as_index=False)
              .head(1)
        )
    else:
        day_best = df.sort_values(["entry_date"], ascending=True)

    day_best = day_best.sort_values("entry_date").reset_index(drop=True)

    # Enforce "one position at a time"
    taken_rows = []
    next_free_date = pd.Timestamp.min
    for _, row in day_best.iterrows():
        if row["entry_date"] >= next_free_date:
            taken_rows.append(row)
            # next trade can only start after this one exits
            next_free_date = row["exit_date"]

    taken = pd.DataFrame(taken_rows)
    if taken.empty:
        empty_curve = pd.DataFrame({"date": [], "equity": []})
        return PolicyAResult(
            trades=taken,
            equity_curve=empty_curve,
            total_R=0.0,
            avg_R=0.0,
            final_equity=initial_equity,
            max_drawdown_pct=0.0,
        )

    # Equity curve using R * risk%
    equity = initial_equity
    curve_dates = []
    curve_equity = []

    total_R = float(taken["R"].sum())
    avg_R = float(taken["R"].mean())

    for _, row in taken.iterrows():
        curve_dates.append(row["exit_date"])
        equity = equity * (1.0 + float(row["R"]) * risk_per_trade_pct)
        curve_equity.append(equity)

    equity_curve = pd.DataFrame({"date": curve_dates, "equity": curve_equity}).sort_values("date")
    equity_curve = equity_curve.reset_index(drop=True)

    # Max drawdown
    running_peak = equity_curve["equity"].cummax()
    drawdown = (equity_curve["equity"] - running_peak) / running_peak
    max_dd_pct = float(drawdown.min()) if not drawdown.empty else 0.0

    return PolicyAResult(
        trades=taken,
        equity_curve=equity_curve,
        total_R=total_R,
        avg_R=avg_R,
        final_equity=float(equity),
        max_drawdown_pct=max_dd_pct,
    )


def main() -> None:
    # Project root = folder containing this file's parent (project_purple/)
    project_root = Path(__file__).resolve().parents[1]

    # ---- 1) Market regime ----
    benchmark = "SPY"  # TODO later: make configurable in one place
    state = get_market_state(benchmark)
    print("\n=== MARKET REGIME ===")
    print(f"Benchmark : {benchmark}")
    print(f"Regime    : {state.regime}")
    print(f"TradeLong : {state.trade_long}")
    print(f"Risk mult : {state.risk_multiplier:.2f}x")

    # ---- 2) Risk config (IMPORTANT: NO atr_period HERE) ----
    base_risk_per_trade_pct = 0.01  # 0.01 = 1% (decimal fraction)
    risk_per_trade_pct = base_risk_per_trade_pct * float(state.risk_multiplier)

    risk_config = RiskConfig(
        risk_per_trade_pct=risk_per_trade_pct,
        atr_stop_multiple=1.5,
        atr_target_multiple=3.0,
        max_position_pct_of_equity=0.20,  # 0.20 = 20% (decimal fraction)
        min_shares=1,
    )

    print(f"Risk/trade: {risk_config.risk_per_trade_pct:.3%} (base {base_risk_per_trade_pct:.3%})")

    if not state.trade_long:
        print("\nRegime says NO LONG trades right now. Exiting (no backtest run).")
        return

    # ---- 3) Universe ----
    build_universe = _safe_import_build_universe()

    if build_universe is not None:
        try:
            universe_symbols = list(build_universe())
        except TypeError:
            # If your build_universe signature changed later, we'll fall back.
            universe_symbols = []
    else:
        universe_symbols = []

    if not universe_symbols:
        # Fallback list (safe default)
        universe_symbols = [
            "AAPL", "AMD", "AMDL", "AMZN", "IBKR",
            "META", "MSFT", "NVDA", "RUN", "SPY", "TECL", "TSLA"
        ]

    print("\nUniverse:")
    print(universe_symbols)

    # ---- 4) Backtest per symbol ----
    initial_equity = 100_000.0
    max_hold_days = 10
    trail_atr_multiple = 3.0

    all_trades: List[pd.DataFrame] = []
    for symbol in universe_symbols:
        print(f"\n===== BACKTESTING {symbol} =====")
        trades_df, _final_equity = run_backtest_for_symbol(
            symbol=symbol,
            project_root=project_root,
            risk_config=risk_config,
            initial_equity=initial_equity,
            max_hold_days=max_hold_days,
            trail_atr_multiple=trail_atr_multiple,
        )
        if trades_df is not None and not trades_df.empty:
            all_trades.append(trades_df)

    if not all_trades:
        print("\nNo trades generated across universe.")
        return

    combined = pd.concat(all_trades, ignore_index=True)

    # ---- 5) Combined analysis ----
    # backtest_v2.analyze_combined_trades() does NOT take initial_equity
    analyze_combined_trades(combined)

    # ---- 6) Portfolio Policy A ----
    print("\n===== PORTFOLIO POLICY A =====")
    policyA = apply_portfolio_policy_A(
        combined_trades=combined,
        initial_equity=initial_equity,
        risk_per_trade_pct=risk_config.risk_per_trade_pct,
    )
    print(f"Trades        : {len(policyA.trades)}")
    print(f"Total R       : {policyA.total_R:.2f}R")
    print(f"Avg R / trade : {policyA.avg_R:.2f}R")
    print(f"Final equity  : ${policyA.final_equity:,.0f}")
    print(f"Max drawdown  : {policyA.max_drawdown_pct:.2%}")


if __name__ == "__main__":
    main()
