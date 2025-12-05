# project_purple/main.py

from __future__ import annotations

from project_purple.data_loader import load_history
from project_purple.indicators import add_basic_trend_indicators
from project_purple.signals import add_basic_long_signal
from project_purple.backtest import run_simple_backtest


def main() -> None:
    # Starting account size for the test
    initial_equity = 100_000.0
    equity = initial_equity

    # Simple universe for now
    symbols = ["AAPL", "NVDA"]

    for symbol in symbols:
        # 1) Load history (from CSV for now)
        df = load_history(symbol, source="csv")

        # 2) Add indicators (EMAs, ATR, 20d high)
        df = add_basic_trend_indicators(df)

        # 3) Add long-entry signal
        df = add_basic_long_signal(df)

        # 4) Run simple backtest for this symbol
        trades_df, equity = run_simple_backtest(df, symbol, initial_equity=equity)

        print("=" * 80)
        print(f"{symbol} trades")

        if trades_df.empty:
            print("No trades generated.\n")
        else:
            # Choose key columns to display
            cols = [
                "symbol",
                "entry_date",
                "entry_price",
                "exit_date",
                "exit_price",
                "shares",
                "risk_dollars",
                "pnl",
                "R",
                "exit_reason",
                "equity_after",
            ]
            print(trades_df[cols], "\n")

            # Simple summary stats for this symbol
            win_rate = (trades_df["R"] > 0).mean() if not trades_df.empty else 0.0
            avg_R = trades_df["R"].mean() if not trades_df.empty else 0.0

            print(
                f"Summary for {symbol}: "
                f"{len(trades_df)} trades, "
                f"win rate: {win_rate:.0%}, "
                f"avg R: {avg_R:.2f}"
            )
            print()

        print(f"Equity after {symbol}: {equity:,.2f}\n")

    print("=" * 80)
    print(f"Starting equity: {initial_equity:,.2f}")
    print(f"Final equity:    {equity:,.2f}")


if __name__ == "__main__":
    main()
