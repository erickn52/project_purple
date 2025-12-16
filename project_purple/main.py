# project_purple/main.py

from __future__ import annotations

from pathlib import Path
from typing import List

from risk import RiskConfig
from scanner_simple import build_universe
from backtest_v2 import (
    run_backtest_for_symbol,
    analyze_combined_trades,
)
from market_state import get_market_state

# Console colors for pretty printing
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def apply_portfolio_policy_A(combined_trades):
    """
    Policy A (start simple):
      - At most 1 open trade at a time (global).
      - If multiple trades share the same entry_date, take the one with the best entry_score.
      - Trades are executed sequentially: next trade starts only after current trade exits.

    Requirements:
      combined_trades must include:
        - entry_date, exit_date, symbol
        - entry_score (recommended; if missing, falls back to 0)
    """
    import pandas as pd

    if combined_trades.empty:
        return combined_trades

    df = combined_trades.copy()

    # Ensure datetime
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])

    # Entry score safety
    if "entry_score" not in df.columns:
        df["entry_score"] = 0.0
    df["entry_score"] = df["entry_score"].fillna(0.0)

    # Best trade per day (by entry_score)
    df_best_per_day = (
        df.sort_values(["entry_date", "entry_score"], ascending=[True, False])
        .groupby("entry_date", as_index=False)
        .first()
    )

    # Enforce one trade at a time
    selected = []
    current_exit = None

    for _, tr in df_best_per_day.iterrows():
        if current_exit is None:
            selected.append(tr)
            current_exit = tr["exit_date"]
        else:
            # accept only if this trade starts AFTER the last one ended
            if tr["entry_date"] > current_exit:
                selected.append(tr)
                current_exit = tr["exit_date"]

    return pd.DataFrame(selected)


def main() -> None:
    """
    Project Purple main entry point.

    Responsibilities:
      - Check overall market regime (SPY).
      - Build today's trading universe using scanner_simple.build_universe().
      - For each symbol in the universe:
          * Load its CSV data
          * Build indicators & momentum/pullback signals
          * Run the ATR/R-based backtest engine with optional trailing stop
          * Print per-symbol summary
      - Combine all trades and print portfolio-level stats and R-distribution.
      - Apply portfolio Policy A to simulate a “1-at-a-time” portfolio.
    """

    # --------------------------------------------------------
    # 1. Basic configuration
    # --------------------------------------------------------
    project_root = Path(__file__).resolve().parents[1]

    # Risk configuration for all symbols
    risk_config = RiskConfig(
        risk_per_trade_pct=0.01,         # risk 1% of account per trade
        atr_stop_multiple=1.5,           # stop 1.5 ATR below entry
        atr_target_multiple=3.0,         # target 3 ATR above entry
        max_position_pct_of_equity=0.20, # max 20% of equity in a single stock
        min_shares=1,
    )

    initial_equity = 100_000.0
    max_hold_days = 10

    # Optional ATR trailing stop.
    #   - Set to None to disable trailing (fixed stop/target only).
    #   - Set to a value like 2.0 or 3.0 to enable trailing.
    trail_atr_multiple: float | None = 3.0

    # --------------------------------------------------------
    # 2. Market regime check (SPY)
    # --------------------------------------------------------
    try:
        market_state = get_market_state(symbol="SPY")
        print("\n===== MARKET REGIME (SPY) =====")
        print(f"As of date : {market_state.as_of_date}")

        # Colorful regime string
        if market_state.regime == "BULL":
            regime_str = f"{GREEN}{market_state.regime}{RESET}"
        elif market_state.regime == "BEAR":
            regime_str = f"{RED}{market_state.regime}{RESET}"
        else:
            regime_str = f"{YELLOW}{market_state.regime}{RESET}"

        trade_long_str = (
            f"{GREEN}True{RESET}"
            if market_state.trade_long
            else f"{RED}False{RESET}"
        )

        print(f"Regime     : {regime_str}")
        print(f"Trade long : {trade_long_str}")
        print(f"Close      : {market_state.close:.2f}")
        print(f"50 MA      : {market_state.ma_fast:.2f}")
        print(f"200 MA     : {market_state.ma_slow:.2f}")
        if market_state.atr is not None:
            print(f"ATR(14)    : {market_state.atr:.2f}")
        else:
            print("ATR(14)    : N/A")

        # Apply regime-based risk multiplier (BULL=1.0, CHOPPY=0.5, BEAR=0.0)
        # Scanner already enforces stricter selection in CHOPPY and returns an empty universe in BEAR.
        if market_state.regime == "BULL":
            regime_risk_mult = 1.0
        elif market_state.regime == "CHOPPY":
            regime_risk_mult = 0.5
        elif market_state.regime == "BEAR":
            regime_risk_mult = 0.0
        else:
            regime_risk_mult = 1.0

        base_risk_pct = float(risk_config.risk_per_trade_pct)
        risk_config.risk_per_trade_pct = base_risk_pct * regime_risk_mult

        print(f"Risk mult  : {regime_risk_mult:.2f}x")
        print(f"Risk/trade : {risk_config.risk_per_trade_pct:.3%} (base {base_risk_pct:.3%})")
    except Exception as e:
        print("\nWARNING: Could not determine market regime from SPY.")
        print(f"Reason: {e}")
        market_state = None
        print("Risk mult  : N/A")
        print(f"Risk/trade : {risk_config.risk_per_trade_pct:.3%} (base, regime unknown)")

    # --------------------------------------------------------
    # 3. Build trading universe
    # --------------------------------------------------------
    print("\n===== BUILDING TRADING UNIVERSE =====")
    universe: List[str] = build_universe()

    if not universe:
        print("\nNo symbols passed the scanner filters. Nothing to backtest.")
        return

    print(f"\nUniverse size: {len(universe)}")
    print(f"Symbols      : {universe}")

    # --------------------------------------------------------
    # 4. Run backtest per symbol
    # --------------------------------------------------------
    all_trades = []

    for symbol in universe:
        print(f"\n\n========== RUNNING BACKTEST FOR {symbol} ==========")
        try:
            trades_df, _final_equity = run_backtest_for_symbol(
                symbol=symbol,
                project_root=project_root,
                risk_config=risk_config,
                initial_equity=initial_equity,
                max_hold_days=max_hold_days,
                trail_atr_multiple=trail_atr_multiple,
            )

            if trades_df.empty:
                print(f"No trades for {symbol}.")
                continue

            # Attach symbol column if needed
            if "symbol" not in trades_df.columns:
                trades_df["symbol"] = symbol

            all_trades.append(trades_df)

            # Print summary (first few rows)
            print(trades_df.head(10))

        except Exception as e:
            print(f"ERROR: Backtest failed for {symbol}: {e}")

    if not all_trades:
        print("\nNo trades produced across the universe.")
        return

    # --------------------------------------------------------
    # 5. Combine trades and analyze portfolio-wide performance
    # --------------------------------------------------------
    import pandas as pd

    combined_trades = pd.concat(all_trades, ignore_index=True)

    print("\n\n===== COMBINED TRADES (ALL SYMBOLS) =====")
    print(combined_trades.head(20))

    print("\n\n===== COMBINED PORTFOLIO SUMMARY =====")
    analyze_combined_trades(combined_trades)

    # --------------------------------------------------------
    # 6. Policy A: One trade at a time
    # --------------------------------------------------------
    policy_trades = apply_portfolio_policy_A(combined_trades)

    print("\n\n===== PORTFOLIO POLICY A (1-TRADE-AT-A-TIME) – FIRST 10 ROWS =====")
    if policy_trades.empty:
        print("No trades selected under Policy A.")
        return

    print(policy_trades.head(10))

    print("\n\n===== POLICY A SUMMARY =====")
    analyze_combined_trades(policy_trades)


if __name__ == "__main__":
    main()
