from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from data_loader import load_symbol_daily
from scanner_simple import build_universe


# ----------------------- Shared strategy settings ----------------------------

INITIAL_EQUITY = 10_000.0

# Moving average parameters
FAST_MA = 20
SLOW_MA = 50

# ATR-based risk management
ATR_WINDOW = 14
ATR_STOP_MULTIPLE = 3.0       # stop distance = ATR * 3
RISK_PER_TRADE = 0.0075       # 0.75% of equity per trade

# Time exit (swing-trade length)
MAX_HOLD_BARS = 10

# Slippage per side (0.05% = 0.10% round trip)
SLIPPAGE_PCT = 0.0005

# Portfolio-level controls
MAX_POSITIONS = 5             # max open trades at once


# ----------------------------- Data structures -------------------------------

@dataclass
class Position:
    symbol: str
    entry_date: pd.Timestamp
    entry_index: int           # bar index where we entered
    entry_price: float         # filled price (with slippage)
    shares: int
    stop_level: float          # raw price (no slippage)
    risk_dollars: float


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
    exit_reason: str           # "stop", "time", "signal", "signal_end"
    equity_after: float        # portfolio equity after closing trade


# ----------------------------- Indicators ------------------------------------

def add_ma_and_atr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Same indicator logic as in atr_backtest.py:
      - ma_fast, ma_slow
      - signal (1 when ma_fast > ma_slow)
      - atr
    """
    df = df.copy()

    df["ma_fast"] = df["close"].rolling(window=FAST_MA, min_periods=FAST_MA).mean()
    df["ma_slow"] = df["close"].rolling(window=SLOW_MA, min_periods=SLOW_MA).mean()
    df["signal"] = (df["ma_fast"] > df["ma_slow"]).astype(int)

    prev_close = df["close"].shift(1)
    high_low = df["high"] - df["low"]
    high_prev = (df["high"] - prev_close).abs()
    low_prev = (df["low"] - prev_close).abs()
    tr = pd.concat([high_low, high_prev, low_prev], axis=1).max(axis=1)
    df["atr"] = tr.rolling(window=ATR_WINDOW, min_periods=ATR_WINDOW).mean()

    return df


# -------------------------- Portfolio backtest -------------------------------

def run_portfolio_backtest() -> Dict:
    """
    Portfolio-level backtest:
      - Multiple symbols traded simultaneously
      - Same MA/ATR/time/stop/slippage logic as single-symbol backtest
      - Uses dynamic universe from scanner_simple.build_universe()
      - Uses fixed max number of positions and risk-per-trade sizing
    """

    # 1) Build today's universe and load data for each symbol
    universe = build_universe()
    if not universe:
        print("Universe is empty. Nothing to backtest.")
        return {}

    print(f"\nPortfolio backtest universe ({len(universe)} symbols): {universe}\n")

    symbol_dfs: Dict[str, pd.DataFrame] = {}
    for sym in universe:
        df_raw = load_symbol_daily(sym)
        if df_raw.empty:
            print(f"WARNING: no data for {sym}, skipping.")
            continue
        symbol_dfs[sym] = add_ma_and_atr(df_raw)

    if not symbol_dfs:
        print("No symbol data loaded. Aborting.")
        return {}

    # Assume all symbols share the same calendar; use first one as driver
    first_sym = next(iter(symbol_dfs.keys()))
    base_df = symbol_dfs[first_sym]
    num_bars = len(base_df)
    dates = pd.to_datetime(base_df["date"]).tolist()

    # 2) Portfolio state
    cash = INITIAL_EQUITY
    equity = INITIAL_EQUITY

    positions: Dict[str, Position] = {}
    trades: List[Trade] = []
    equity_series: List[float] = []
    date_series: List[pd.Timestamp] = []

    # 3) Iterate over each bar (day)
    for i in range(num_bars):
        date = dates[i]

        # ---------------- EXIT LOGIC for all open positions -------------------
        symbols_to_close: List[str] = []

        for sym, pos in list(positions.items()):
            df_sym = symbol_dfs[sym]
            if i >= len(df_sym):
                continue  # symbol has no more data

            row = df_sym.iloc[i]
            close = float(row["close"])
            low = float(row["low"])
            signal_today = int(row["signal"])
            signal_yesterday = int(df_sym.iloc[i - 1]["signal"]) if i > 0 else 0

            exit_reason = None
            raw_exit_price = None

            # 1) Stop exit
            if low <= pos.stop_level:
                exit_reason = "stop"
                raw_exit_price = pos.stop_level

            # 2) Time exit (if no stop)
            elif (i - pos.entry_index) >= MAX_HOLD_BARS:
                exit_reason = "time"
                raw_exit_price = close

            # 3) Signal exit (if no stop/time)
            elif signal_yesterday == 1 and signal_today == 0:
                exit_reason = "signal"
                raw_exit_price = close

            if exit_reason is not None:
                exit_price = raw_exit_price * (1.0 - SLIPPAGE_PCT)
                pnl = (exit_price - pos.entry_price) * pos.shares
                R = pnl / pos.risk_dollars if pos.risk_dollars else 0.0

                cash += pos.shares * exit_price
                symbols_to_close.append(sym)

                # We'll update equity after exits for the day
                # (value of remaining positions will be added below)
                # For now, mark equity as cash; we'll add positions back.
                # We'll compute final equity for the trade after we mark-to-market.
                # For clarity, we compute an approximate equity_after using
                # today's closing prices.

                # Temporarily remove, we'll rebuild equity after the loop.
                # We'll push the Trade object after we recompute equity.
                trade = Trade(
                    symbol=sym,
                    entry_date=pos.entry_date,
                    exit_date=date,
                    entry_price=pos.entry_price,
                    exit_price=exit_price,
                    shares=pos.shares,
                    pnl=pnl,
                    return_pct=(exit_price / pos.entry_price - 1.0) * 100.0,
                    R=R,
                    exit_reason=exit_reason,
                    equity_after=0.0,  # placeholder, set later
                )
                trades.append(trade)

        # Remove closed positions from dict
        for sym in symbols_to_close:
            positions.pop(sym, None)

        # ---------------- ENTRY LOGIC (after exits) --------------------------
        # Determine available slots
        slots_left = MAX_POSITIONS - len(positions)
        if slots_left > 0:
            # Gather all potential new entries
            candidates: List[str] = []

            for sym, df_sym in symbol_dfs.items():
                if sym in positions:
                    continue
                if i == 0 or i >= len(df_sym):
                    continue

                row_today = df_sym.iloc[i]
                row_yest = df_sym.iloc[i - 1]

                signal_today = int(row_today["signal"])
                signal_yesterday = int(row_yest["signal"])

                if signal_yesterday == 0 and signal_today == 1:
                    candidates.append(sym)

            # Simple selection rule: alphabetical order (placeholder).
            # Later we can rank by momentum / volatility / sector, etc.
            candidates.sort()

            for sym in candidates[:slots_left]:
                df_sym = symbol_dfs[sym]
                row = df_sym.iloc[i]
                close = float(row["close"])
                atr = float(row["atr"]) if not pd.isna(row["atr"]) else None

                if atr is None or atr <= 0:
                    continue

                risk_per_share = ATR_STOP_MULTIPLE * atr
                equity = cash + sum(
                    p.shares * float(symbol_dfs[s]["close"].iloc[i])
                    for s, p in positions.items()
                )
                risk_dollars = equity * RISK_PER_TRADE
                shares_to_buy = int(risk_dollars // risk_per_share)

                if shares_to_buy <= 0:
                    continue

                raw_entry_price = close
                entry_price = raw_entry_price * (1.0 + SLIPPAGE_PCT)
                cost = shares_to_buy * entry_price

                if cost > cash:
                    continue  # not enough cash for this position

                stop_level = raw_entry_price - risk_per_share

                pos = Position(
                    symbol=sym,
                    entry_date=date,
                    entry_index=i,
                    entry_price=entry_price,
                    shares=shares_to_buy,
                    stop_level=stop_level,
                    risk_dollars=risk_dollars,
                )

                positions[sym] = pos
                cash -= cost

        # ---------------- Mark-to-market portfolio equity --------------------
        equity = cash
        for sym, pos in positions.items():
            df_sym = symbol_dfs[sym]
            if i < len(df_sym):
                close = float(df_sym.iloc[i]["close"])
                equity += pos.shares * close

        # Now that equity is known, update equity_after on today's trades
        if trades:
            # Find trades that closed today and still have equity_after == 0.0
            for t in trades:
                if t.exit_date == date and t.equity_after == 0.0:
                    t.equity_after = equity

        equity_series.append(equity)
        date_series.append(date)

    # ---------------------- Final stats & outputs ----------------------------

    equity_arr = np.array(equity_series, dtype=float)
    final_equity = float(equity_arr[-1])
    total_return_pct = (final_equity / INITIAL_EQUITY - 1.0) * 100.0

    running_max = np.maximum.accumulate(equity_arr)
    drawdowns = equity_arr / running_max - 1.0
    max_drawdown_pct = float(drawdowns.min() * 100.0)

    trades_df = pd.DataFrame([t.__dict__ for t in trades]) if trades else pd.DataFrame()

    if not trades_df.empty:
        win_rate = (trades_df["pnl"] > 0).mean() * 100.0
        avg_R = trades_df["R"].mean()
        num_trades = len(trades_df)
    else:
        win_rate = 0.0
        avg_R = 0.0
        num_trades = 0

    stats = {
        "initial_equity": INITIAL_EQUITY,
        "final_equity": final_equity,
        "total_return_pct": total_return_pct,
        "num_trades": num_trades,
        "win_rate_pct": win_rate,
        "avg_R": avg_R,
        "max_drawdown_pct": max_drawdown_pct,
    }

    # Print a short report
    print("\n=== Portfolio Summary (MA + ATR + time + slippage) ===")
    for k, v in stats.items():
        print(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}")

    if not trades_df.empty:
        print("\n=== First 10 trades ===")
        print(trades_df.head(10).to_string(index=False))

    equity_df = pd.DataFrame({"date": date_series, "equity": equity_series})
    print("\n=== Portfolio equity (last 10 rows) ===")
    print(equity_df.tail(10).to_string(index=False))

    return {
        "stats": stats,
        "trades": trades_df,
        "equity": equity_df,
    }


# ------------------------------ Main entry -----------------------------------

if __name__ == "__main__":
    run_portfolio_backtest()
