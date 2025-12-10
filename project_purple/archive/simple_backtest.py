from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import pandas as pd

# Direct imports because we run as a script inside the package folder
from data_loader import load_symbol_daily, TICKERS


INITIAL_EQUITY = 10_000.0
FAST_MA = 20
SLOW_MA = 50


@dataclass
class Trade:
    symbol: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    return_pct: float


def compute_moving_average_signals(
    df: pd.DataFrame,
    fast: int = FAST_MA,
    slow: int = SLOW_MA,
) -> pd.DataFrame:
    """
    Adds moving-average based signals and position to the DataFrame.

    Rules (long-only):
      - When fast MA > slow MA => long (position = 1)
      - When fast MA <= slow MA => flat (position = 0)
      - We enter/exit on the NEXT day's bar by shifting the signal by 1.
    """
    df = df.copy()

    df["ma_fast"] = df["close"].rolling(window=fast, min_periods=fast).mean()
    df["ma_slow"] = df["close"].rolling(window=slow, min_periods=slow).mean()

    # Signal: 1 when fast > slow, else 0
    df["signal"] = 0
    df.loc[df["ma_fast"] > df["ma_slow"], "signal"] = 1

    # Position: yesterday's signal (enter next bar)
    df["position"] = df["signal"].shift(1).fillna(0)

    return df


def backtest_single_symbol(
    df: pd.DataFrame,
    symbol: str,
    initial_equity: float = INITIAL_EQUITY,
) -> Dict:
    """
    Vectorized backtest for a single symbol using the MA strategy.

    - Fully invested when position = 1, in cash when position = 0.
    - Equity curve is computed from daily returns.
    - Also extracts a list of individual trades.
    """
    df = compute_moving_average_signals(df)

    # Simple daily returns on close-to-close
    df["return"] = df["close"].pct_change().fillna(0.0)

    # Strategy returns: only earn returns when in a position
    df["strategy_return"] = df["position"] * df["return"]

    # Equity curve
    df["equity"] = initial_equity * (1.0 + df["strategy_return"]).cumprod()

    # Build list of trades from position changes
    trades: List[Trade] = []
    in_trade = False
    entry_price = 0.0
    entry_date = None

    prev_pos = 0.0

    for _, row in df.iterrows():
        pos = row["position"]
        price = row["close"]
        date = row["date"]

        # Enter trade: position goes from 0 -> 1
        if (not in_trade) and prev_pos == 0 and pos == 1:
            in_trade = True
            entry_price = float(price)
            entry_date = pd.to_datetime(date)

        # Exit trade: position goes from 1 -> 0
        elif in_trade and prev_pos == 1 and pos == 0:
            exit_price = float(price)
            exit_date = pd.to_datetime(date)
            ret = (exit_price / entry_price) - 1.0
            trades.append(
                Trade(
                    symbol=symbol,
                    entry_date=entry_date,
                    exit_date=exit_date,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    return_pct=ret * 100.0,
                )
            )
            in_trade = False

        prev_pos = pos

    # If still in a trade at the end, close it on the last bar
    if in_trade:
        last_row = df.iloc[-1]
        exit_price = float(last_row["close"])
        exit_date = pd.to_datetime(last_row["date"])
        ret = (exit_price / entry_price) - 1.0
        trades.append(
            Trade(
                symbol=symbol,
                entry_date=entry_date,
                exit_date=exit_date,
                entry_price=entry_price,
                exit_price=exit_price,
                return_pct=ret * 100.0,
            )
        )

    trades_df = (
        pd.DataFrame([t.__dict__ for t in trades])
        if trades
        else pd.DataFrame(
            columns=[
                "symbol",
                "entry_date",
                "exit_date",
                "entry_price",
                "exit_price",
                "return_pct",
            ]
        )
    )

    # Basic stats
    final_equity = float(df["equity"].iloc[-1])
    total_return_pct = (final_equity / initial_equity - 1.0) * 100.0

    if not trades_df.empty:
        num_trades = len(trades_df)
        wins = (trades_df["return_pct"] > 0).sum()
        win_rate = wins / num_trades * 100.0

        # Max drawdown
        equity = df["equity"].values
        running_max = np.maximum.accumulate(equity)
        drawdowns = equity / running_max - 1.0
        max_drawdown_pct = float(drawdowns.min() * 100.0)
    else:
        num_trades = 0
        win_rate = 0.0
        max_drawdown_pct = 0.0

    stats = {
        "symbol": symbol,
        "initial_equity": initial_equity,
        "final_equity": final_equity,
        "total_return_pct": total_return_pct,
        "num_trades": num_trades,
        "win_rate_pct": win_rate,
        "max_drawdown_pct": max_drawdown_pct,
    }

    return {
        "df": df,
        "trades": trades_df,
        "stats": stats,
    }


def run_example():
    """
    Simple demo backtest on AAPL using the historical data you downloaded.
    """
    symbol = "AAPL"
    print(f"Running simple MA backtest on {symbol}...")

    df = load_symbol_daily(symbol)
    result = backtest_single_symbol(df, symbol, INITIAL_EQUITY)

    stats = result["stats"]
    trades = result["trades"]

    print("\n=== Summary Stats ===")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"{k}: {v:.2f}")
        else:
            print(f"{k}: {v}")

    print("\n=== First 5 trades ===")
    if trades.empty:
        print("No trades generated by this strategy.")
    else:
        print(trades.head())

    print("\n=== Equity curve (last 5 rows) ===")
    equity_tail = result["df"][["date", "equity"]].tail()
    print(equity_tail)


def run_all_symbols():
    """
    Run the same MA backtest on all tickers in TICKERS
    and print a summary table of performance.
    """
    summary_rows: List[Dict] = []

    print("\n\nRunning MA backtest on all symbols in TICKERS...")
    print(f"Universe: {TICKERS}\n")

    for symbol in TICKERS:
        try:
            df = load_symbol_daily(symbol)
            result = backtest_single_symbol(df, symbol, INITIAL_EQUITY)
            stats = result["stats"]
            summary_rows.append(stats)
        except Exception as e:
            print(f"WARNING: Failed to backtest {symbol}: {e}")

    if not summary_rows:
        print("No results to show.")
        return

    summary_df = pd.DataFrame(summary_rows)

    # Focus columns for readability
    cols = [
        "symbol",
        "total_return_pct",
        "num_trades",
        "win_rate_pct",
        "max_drawdown_pct",
    ]
    summary_df = summary_df[cols].sort_values(
        "total_return_pct", ascending=False
    )

    print("\n=== Summary Across Symbols (sorted by total_return_pct) ===")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    run_example()
    run_all_symbols()
