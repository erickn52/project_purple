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

        # Trade long: green if True, red if False
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
    except Exception as e:
        print("\nWARNING: Could not determine market regime from SPY.")
        print(f"Reason: {e}")
        market_state = None

    # (Note: we no longer use a separate 'no_trade' flag here.
    #  Trade-long behavior is described by market_state.trade_long
    #  and can be wired into live trading later.)

    # --------------------------------------------------------
    # 3. Build trading universe
    # --------------------------------------------------------
    print("\n===== BUILDING TRADING UNIVERSE =====")
    universe: List[str] = build_universe()

    if not universe:
        print("\nNo symbols passed the scanner filters. Nothing to backtest.")
        return

    # If you want to limit the number of symbols for speed, slice here.
    # Example: universe = universe[:10]
    print(f"\nUniverse size: {len(universe)}")
    print(f"Symbols      : {universe}")

    # --------------------------------------------------------
    # 4. Run backtest per symbol
    # --------------------------------------------------------
    all_trades = []

    for symbol in universe:
        print(f"\n\n========== RUNNING BACKTEST FOR {symbol} ==========")
        try:
            trades_df, final_equity = run_backtest_for_symbol(
                symbol=symbol,
                project_root=project_root,
                risk_config=risk_config,
                initial_equity=initial_equity,
                max_hold_days=max_hold_days,
                trail_atr_multiple=trail_atr_multiple,
            )
        except FileNotFoundError as e:
            # Data not downloaded yet for this symbol
            print(f"SKIPPING {symbol}: {e}")
            continue
        except Exception as e:
            # Any other unexpected error
            print(f"ERROR while backtesting {symbol}: {e}")
            continue

        if not trades_df.empty:
            all_trades.append(trades_df)

    # --------------------------------------------------------
    # 5. Combined portfolio-level analysis
    # --------------------------------------------------------
    if all_trades:
        import pandas as pd

        combined_trades = pd.concat(all_trades, ignore_index=True)
        print("\n\n===== COMBINED TRADES (ALL SYMBOLS) – FIRST 10 ROWS =====")
        print(combined_trades.head(10))

        analyze_combined_trades(combined_trades)
    else:
        print("\nNo trades were generated for any symbol in the universe.")


if __name__ == "__main__":
    main()
