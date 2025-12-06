# project_purple/main.py

from __future__ import annotations

from project_purple.data_loader import load_history
from project_purple.indicators import add_basic_trend_indicators
from project_purple.signals import add_basic_long_signal
from project_purple.backtest import run_simple_backtest
from project_purple.settings import strategy_settings


def main() -> None:
    initial_equity = 100_000.0
    equity = initial_equity

    symbols = ["AAPL", "NVDA"]

    # 1) Load market index (SPY) for regime filter, if enabled
    if strategy_settings.use_market_regime:
        try:
            spy = load_history("SPY", source="csv")
            spy = add_basic_trend_indicators(spy)

            # Simple regime rule: uptrend in SPY
            spy["market_long_ok"] = (spy["ema_20"] > spy["ema_50"]) & (
                spy["close"] > spy["ema_20"]
            )

            regime = spy[["market_long_ok"]]
        except FileNotFoundError:
            print(
                "WARNING: SPY data not found in data/. "
                "Regime filter will be disabled for this run."
            )
            regime = None
    else:
        regime = None

    # 2) Run the strategy per symbol
    for symbol in symbols:
        df = load_history(symbol, source="csv")
        df = add_basic_trend_indicators(df)

        # attach regime info if available
        if regime is not None:
            df = df.join(regime, how="left")

        df = add_basic_long_signal(df)

        trades_df, equity = run_simple_backtest(df, symbol, initial_equity=equity)

        print("=" * 80)
        print(f"{symbol} trades")

        if trades_df.empty:
            print("No trades generated.\n")
        else:
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
