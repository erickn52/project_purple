from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import pandas as pd

# Direct import because we run this script from inside the package folder
from data_loader import load_symbol_daily
from scanner_simple import build_universe  # dynamic universe


# --- Risk & indicator settings ------------------------------------------------

INITIAL_EQUITY = 10_000.0

# Moving average parameters
FAST_MA = 20
SLOW_MA = 50

# ATR-based risk management
ATR_WINDOW = 14
ATR_STOP_MULTIPLE = 3.0       # stop distance = ATR * 3
RISK_PER_TRADE = 0.0075       # 0.75% of equity per trade

# TIME EXIT: max bars allowed in a trade
MAX_HOLD_BARS = 10

# SLIPPAGE: assumed per-side slippage as percent of price
# 0.0005 = 0.05% per side (≈0.10% round trip)
SLIPPAGE_PCT = 0.0005


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
    exit_reason: str  # "stop", "signal", "time", "signal_end"


# --- Indicator / signal calculations -----------------------------------------


def add_ma_and_atr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add MA(FAST_MA), MA(SLOW_MA), signal, and ATR(ATR_WINDOW).
    """
    df = df.copy()

    # Moving averages
    df["ma_fast"] = df["close"].rolling(window=FAST_MA, min_periods=FAST_MA).mean()
    df["ma_slow"] = df["close"].rolling(window=SLOW_MA, min_periods=SLOW_MA).mean()

    # Signal: long when fast > slow
    df["signal"] = (df["ma_fast"] > df["ma_slow"]).astype(int)

    # ATR
    prev_close = df["close"].shift(1)
    high_low = df["high"] - df["low"]
    high_prev = (df["high"] - prev_close).abs()
    low_prev = (df["low"] - prev_close).abs()

    tr = pd.concat([high_low, high_prev, low_prev], axis=1).max(axis=1)
    df["atr"] = tr.rolling(window=ATR_WINDOW, min_periods=ATR_WINDOW).mean()

    return df


# --- ATR-backed backtest with TIME EXIT + SLIPPAGE ---------------------------


def backtest_single_symbol_atr(
    df: pd.DataFrame,
    symbol: str,
    initial_equity: float = INITIAL_EQUITY,
) -> Dict:

    df = add_ma_and_atr(df)

    trades: List[Trade] = []

    cash = initial_equity
    equity = initial_equity
    shares = 0
    in_trade = False
    entry_price = 0.0        # actual fill (with slippage)
    entry_date = None
    entry_index = None       # bar index when we entered the trade
    stop_level = 0.0         # raw price stop trigger (no slippage)
    risk_dollars = 0.0

    prev_signal = 0
    equity_series: List[float] = []
    date_series: List[pd.Timestamp] = []

    for i, (_, row) in enumerate(df.iterrows()):
        date = pd.to_datetime(row["date"])
        close = float(row["close"])
        low = float(row["low"])
        signal = int(row["signal"])
        atr = float(row["atr"]) if not pd.isna(row["atr"]) else None

        # =============================
        # EXIT LOGIC (if in trade)
        # =============================
        if in_trade and shares > 0:

            # 1) STOP EXIT (trigger uses raw stop_level, fill uses slippage)
            if low <= stop_level:
                raw_exit_price = stop_level
                exit_price = raw_exit_price * (1.0 - SLIPPAGE_PCT)

                cash += shares * exit_price
                pnl = (exit_price - entry_price) * shares
                R = pnl / risk_dollars if risk_dollars else 0.0

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
                in_trade = False
                shares = 0
                entry_index = None

            # 2) TIME EXIT: 10 bars
            elif entry_index is not None and (i - entry_index) >= MAX_HOLD_BARS:
                raw_exit_price = close
                exit_price = raw_exit_price * (1.0 - SLIPPAGE_PCT)

                cash += shares * exit_price
                pnl = (exit_price - entry_price) * shares
                R = pnl / risk_dollars if risk_dollars else 0.0

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
                        exit_reason="time",
                    )
                )
                in_trade = False
                shares = 0
                entry_index = None

            # 3) SIGNAL EXIT
            elif prev_signal == 1 and signal == 0:
                raw_exit_price = close
                exit_price = raw_exit_price * (1.0 - SLIPPAGE_PCT)

                cash += shares * exit_price
                pnl = (exit_price - entry_price) * shares
                R = pnl / risk_dollars if risk_dollars else 0.0

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
                entry_index = None

            else:
                # Still in trade: mark to market with raw close
                equity = cash + shares * close

        else:
            equity = cash

        # =============================
        # ENTRY LOGIC
        # =============================
        if (not in_trade) and prev_signal == 0 and signal == 1:
            if atr and atr > 0:
                risk_per_share = ATR_STOP_MULTIPLE * atr
                risk_dollars = equity * RISK_PER_TRADE
                shares_to_buy = int(risk_dollars // risk_per_share)

                if shares_to_buy > 0:
                    raw_entry_price = close
                    entry_price = raw_entry_price * (1.0 + SLIPPAGE_PCT)
                    entry_date = date
                    entry_index = i

                    # stop_level is set using raw entry price (no slippage)
                    stop_level = raw_entry_price - risk_per_share

                    shares = shares_to_buy
                    cash -= shares * entry_price  # pay slipped price
                    in_trade = True
                    equity = cash + shares * close

        equity_series.append(equity)
        date_series.append(date)
        prev_signal = signal

    # =============================
    # EXIT if trade still open at end
    # =============================
    if in_trade and shares > 0:
        last_date = pd.to_datetime(df.iloc[-1]["date"])
        raw_exit_price = float(df.iloc[-1]["close"])
        exit_price = raw_exit_price * (1.0 - SLIPPAGE_PCT)

        pnl = (exit_price - entry_price) * shares
        R = pnl / risk_dollars if risk_dollars else 0.0
        cash += shares * exit_price

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
        equity_series[-1] = cash

    # =============================
    # RESULTS
    # =============================
    df_result = df.copy()
    df_result["equity"] = equity_series

    trades_df = pd.DataFrame([t.__dict__ for t in trades])

    equity_arr = np.array(equity_series)
    final_equity = float(equity_arr[-1])
    total_return_pct = (final_equity / initial_equity - 1.0) * 100.0

    running_max = np.maximum.accumulate(equity_arr)
    drawdowns = equity_arr / running_max - 1.0
    max_drawdown_pct = float(drawdowns.min() * 100.0)

    if len(trades_df) > 0:
        win_rate = (trades_df["pnl"] > 0).mean() * 100.0
        avg_R = trades_df["R"].mean()
    else:
        win_rate = 0.0
        avg_R = 0.0

    stats = {
        "symbol": symbol,
        "initial_equity": initial_equity,
        "final_equity": final_equity,
        "total_return_pct": total_return_pct,
        "num_trades": len(trades_df),
        "win_rate_pct": win_rate,
        "avg_R": avg_R,
        "max_drawdown_pct": max_drawdown_pct,
    }

    return {"df": df_result, "trades": trades_df, "stats": stats}


# --- RUNNERS -----------------------------------------------------------------


def run_example():
    symbol = "AAPL"
    print(f"\nRunning ATR-based MA backtest on {symbol}...\n")
    df = load_symbol_daily(symbol)
    result = backtest_single_symbol_atr(df, symbol, INITIAL_EQUITY)

    stats = result["stats"]
    trades = result["trades"]

    print("=== Summary Stats (ATR + slippage) ===")
    for k, v in stats.items():
        print(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}")

    print("\n=== First 5 trades (ATR + slippage) ===")
    print(trades.head())

    print("\n=== Equity curve (last 5 rows) ===")
    print(result["df"][["date", "equity"]].tail())


def run_all_symbols():
    print("\n\nRunning ATR-based MA backtest on dynamic universe (with slippage)...\n")
    universe = build_universe()
    print(f"Universe used for backtest: {universe}\n")

    rows = []

    for symbol in universe:
        try:
            df = load_symbol_daily(symbol)
            result = backtest_single_symbol_atr(df, symbol, INITIAL_EQUITY)
            rows.append(result["stats"])
        except Exception as e:
            print(f"WARNING for {symbol}: {e}")

    if not rows:
        print("No results to show.")
        return

    summary = pd.DataFrame(rows)
    cols = ["symbol", "total_return_pct", "num_trades", "win_rate_pct", "avg_R", "max_drawdown_pct"]
    summary = summary[cols].sort_values("total_return_pct", ascending=False)

    print("\n=== ATR-based Summary Across Universe (with slippage) ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    run_example()
    run_all_symbols()
