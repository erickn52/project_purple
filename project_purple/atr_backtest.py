from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import pandas as pd

# Direct import because we run this script from inside the package folder
from data_loader import load_symbol_daily, TICKERS


# --- Risk & indicator settings ------------------------------------------------

INITIAL_EQUITY = 10_000.0

# Moving average parameters (same as simple_backtest for now)
FAST_MA = 20
SLOW_MA = 50

# ATR-based risk management
ATR_WINDOW = 14
ATR_STOP_MULTIPLE = 3.0       # stop distance = ATR * 3
RISK_PER_TRADE = 0.0075       # 0.75% of equity per trade


# --- Data structures ----------------------------------------------------------


@dataclass
class Trade:
    symbol: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    shares: int
    pnl: float
    return_pct: float
    R: float
    exit_reason: str  # "stop" or "signal"


# --- Indicator / signal calculations -----------------------------------------


def add_ma_and_atr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add MA(FAST_MA), MA(SLOW_MA), signal, and ATR(ATR_WINDOW) columns.

    Signal (long-only):
      - signal = 1 when fast MA > slow MA
      - signal = 0 otherwise

    ATR:
      - Classic Wilder-style True Range and simple moving average of TR.
    """
    df = df.copy()

    # Moving averages
    df["ma_fast"] = df["close"].rolling(window=FAST_MA, min_periods=FAST_MA).mean()
    df["ma_slow"] = df["close"].rolling(window=SLOW_MA, min_periods=SLOW_MA).mean()

    # Signal based on MAs
    df["signal"] = 0
    df.loc[df["ma_fast"] > df["ma_slow"], "signal"] = 1

    # ATR
    prev_close = df["close"].shift(1)

    high_low = df["high"] - df["low"]
    high_prev = (df["high"] - prev_close).abs()
    low_prev = (df["low"] - prev_close).abs()

    tr = pd.concat([high_low, high_prev, low_prev], axis=1).max(axis=1)
    df["atr"] = tr.rolling(window=ATR_WINDOW, min_periods=ATR_WINDOW).mean()

    return df


# --- Core backtest with ATR-based position sizing ----------------------------


def backtest_single_symbol_atr(
    df: pd.DataFrame,
    symbol: str,
    initial_equity: float = INITIAL_EQUITY,
) -> Dict:
    """
    Backtest a single symbol using:
      - MA(FAST_MA/SLOW_MA) trend filter for entries/exits
      - ATR-based stop (ATR_STOP_MULTIPLE * ATR)
      - Risk-per-trade sizing (RISK_PER_TRADE of equity)

    Mechanics:
      - If flat and signal flips 0 -> 1:
          * compute ATR
          * risk_per_share = ATR_STOP_MULTIPLE * ATR
          * risk_dollars   = equity * RISK_PER_TRADE
          * shares         = floor(risk_dollars / risk_per_share)
          * enter at close
          * stop_price = entry_price - risk_per_share

      - If in trade:
          * if low <= stop_price -> exit at stop_price (stop)
          * else if signal flips 1 -> 0 -> exit at close (signal)
          * otherwise hold

      - Equity is marked-to-market every bar:
          equity = cash + shares * close
    """
    df = add_ma_and_atr(df)

    trades: List[Trade] = []

    cash = initial_equity
    equity = initial_equity
    shares = 0
    in_trade = False
    entry_price = 0.0
    entry_date = None
    stop_price = 0.0
    risk_dollars = 0.0

    prev_signal = 0
    equity_series: List[float] = []
    date_series: List[pd.Timestamp] = []

    for _, row in df.iterrows():
        date = pd.to_datetime(row["date"])
        close = float(row["close"])
        low = float(row["low"])
        signal = int(row["signal"])
        atr = float(row["atr"]) if not pd.isna(row["atr"]) else None

        # --- Manage open position: check for exit -----------------------------
        if in_trade and shares > 0:
            # 1) Stop hit?
            if low <= stop_price:
                exit_price = stop_price
                cash += shares * exit_price
                pnl = (exit_price - entry_price) * shares
                equity = cash
                R = pnl / risk_dollars if risk_dollars > 0 else 0.0

                trades.append(
                    Trade(
                        symbol=symbol,
                        entry_date=entry_date,
                        exit_date=date,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        shares=shares,
                        pnl=pnl,
                        return_pct=(exit_price / entry_price - 1.0) * 100.0,
                        R=R,
                        exit_reason="stop",
                    )
                )

                # Flat after stop
                in_trade = False
                shares = 0
                entry_price = 0.0
                stop_price = 0.0
                risk_dollars = 0.0

            # 2) Signal-based exit (if no stop triggered)
            elif prev_signal == 1 and signal == 0:
                exit_price = close
                cash += shares * exit_price
                pnl = (exit_price - entry_price) * shares
                equity = cash
                R = pnl / risk_dollars if risk_dollars > 0 else 0.0

                trades.append(
                    Trade(
                        symbol=symbol,
                        entry_date=entry_date,
                        exit_date=date,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        shares=shares,
                        pnl=pnl,
                        return_pct=(exit_price / entry_price - 1.0) * 100.0,
                        R=R,
                        exit_reason="signal",
                    )
                )

                in_trade = False
                shares = 0
                entry_price = 0.0
                stop_price = 0.0
                risk_dollars = 0.0

            else:
                # Still in trade, mark-to-market
                equity = cash + shares * close

        else:
            # Flat: equity is just cash
            equity = cash

        # --- Potential new entry (after exits) --------------------------------
        if (not in_trade) and prev_signal == 0 and signal == 1:
            if atr is not None and atr > 0:
                risk_per_share = ATR_STOP_MULTIPLE * atr
                risk_dollars = equity * RISK_PER_TRADE
                shares_to_buy = int(risk_dollars // risk_per_share)

                if shares_to_buy > 0:
                    entry_price = close
                    entry_date = date
                    stop_price = entry_price - risk_per_share
                    shares = shares_to_buy
                    cash -= shares * entry_price
                    in_trade = True
                    equity = cash + shares * close

        # Record equity for this bar
        equity_series.append(equity)
        date_series.append(date)

        prev_signal = signal

    # --- If still in a trade at the end, close on last close ------------------
    if in_trade and shares > 0:
        # Use last known close and date
        last_close = float(df.iloc[-1]["close"])
        last_date = pd.to_datetime(df.iloc[-1]["date"])

        exit_price = last_close
        cash += shares * exit_price
        pnl = (exit_price - entry_price) * shares
        equity = cash
        R = pnl / risk_dollars if risk_dollars > 0 else 0.0

        trades.append(
            Trade(
                symbol=symbol,
                entry_date=entry_date,
                exit_date=last_date,
                entry_price=entry_price,
                exit_price=exit_price,
                shares=shares,
                pnl=pnl,
                return_pct=(exit_price / entry_price - 1.0) * 100.0,
                R=R,
                exit_reason="signal_end",
            )
        )

        # Update last equity point
        if equity_series:
            equity_series[-1] = equity

    # --- Build DataFrames and stats -------------------------------------------
    df_result = df.copy()
    df_result["equity"] = equity_series

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
                "shares",
                "pnl",
                "return_pct",
                "R",
                "exit_reason",
            ]
        )
    )

    equity_arr = np.array(equity_series, dtype=float)
    final_equity = float(equity_arr[-1])
    total_return_pct = (final_equity / initial_equity - 1.0) * 100.0

    # Max drawdown
    running_max = np.maximum.accumulate(equity_arr)
    drawdowns = equity_arr / running_max - 1.0
    max_drawdown_pct = float(drawdowns.min() * 100.0)

    if not trades_df.empty:
        num_trades = len(trades_df)
        wins = (trades_df["pnl"] > 0).sum()
        win_rate = wins / num_trades * 100.0
        avg_R = trades_df["R"].mean()
    else:
        num_trades = 0
        win_rate = 0.0
        avg_R = 0.0

    stats = {
        "symbol": symbol,
        "initial_equity": initial_equity,
        "final_equity": final_equity,
        "total_return_pct": total_return_pct,
        "num_trades": num_trades,
        "win_rate_pct": win_rate,
        "avg_R": avg_R,
        "max_drawdown_pct": max_drawdown_pct,
    }

    return {
        "df": df_result,
        "trades": trades_df,
        "stats": stats,
    }


# --- Convenience runners ------------------------------------------------------


def run_example():
    symbol = "AAPL"
    print(f"Running ATR-based MA backtest on {symbol}...")

    df = load_symbol_daily(symbol)
    result = backtest_single_symbol_atr(df, symbol, INITIAL_EQUITY)

    stats = result["stats"]
    trades = result["trades"]

    print("\n=== Summary Stats (ATR) ===")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"{k}: {v:.2f}")
        else:
            print(f"{k}: {v}")

    print("\n=== First 5 trades (ATR) ===")
    if trades.empty:
        print("No trades generated by this strategy.")
    else:
        print(trades.head())

    print("\n=== Equity curve (last 5 rows) ===")
    tail = result["df"][["date", "equity"]].tail()
    print(tail)


def run_all_symbols():
    print("\n\nRunning ATR-based MA backtest on all symbols...")
    print(f"Universe: {TICKERS}\n")

    rows: List[Dict] = []

    for symbol in TICKERS:
        try:
            df = load_symbol_daily(symbol)
            result = backtest_single_symbol_atr(df, symbol, INITIAL_EQUITY)
            rows.append(result["stats"])
        except Exception as e:
            print(f"WARNING: Failed for {symbol}: {e}")

    if not rows:
        print("No results to show.")
        return

    summary = pd.DataFrame(rows)

    cols = [
        "symbol",
        "total_return_pct",
        "num_trades",
        "win_rate_pct",
        "avg_R",
        "max_drawdown_pct",
    ]
    summary = summary[cols].sort_values("total_return_pct", ascending=False)

    print("\n=== ATR-based Summary Across Symbols ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    run_example()
    run_all_symbols()
