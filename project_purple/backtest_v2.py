import pandas as pd
from pathlib import Path

from risk import RiskConfig, calculate_risk_for_trade, calculate_R


# ------------------------------------------------------------
# 1. DATA CLEANING & INDICATORS
# ------------------------------------------------------------

def clean_aapl_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the daily CSV format you showed in Excel.

    Original structure:
      Row 1: Price | symbol | open | high | low | close | volume  (headers)
      Row 2: Ticker | XXX   | XXX  | ...                      (metadata)
      Row 3: date | ...                                     (metadata)
      Row 4+: actual data

    Steps:
      - Keep row 1 as header (pandas does this by default).
      - Drop rows where first column is 'Ticker' or 'date'.
      - Rename 'Price' column to 'date'.
      - Lowercase all column names.
    """
    first_col_name = df.columns[0]

    # Drop metadata rows
    df = df[~df[first_col_name].isin(["Ticker", "date"])].reset_index(drop=True)

    # Rename 'Price' -> 'date' if present
    rename_map = {}
    for col in df.columns:
        if col.strip().lower() == "price":
            rename_map[col] = "date"

    if rename_map:
        df = df.rename(columns=rename_map)

    # Standardize column names
    df.columns = [c.strip().lower() for c in df.columns]

    expected_cols = {"date", "symbol", "open", "high", "low", "close", "volume"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"After cleaning, missing expected columns: {missing}. "
            f"Got columns: {list(df.columns)}"
        )

    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Simple ATR(14) calculator using columns:
      open, high, low, close
    """
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")

    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr"] = true_range.rolling(window=period, min_periods=period).mean()

    return df


def add_momentum_pullback_signals(
    df: pd.DataFrame,
    mom_lookback: int = 63,
    trend_short: int = 50,
    trend_long: int = 200,
    pullback_lookback: int = 10,
    min_momentum: float = 0.15,   # +15% over lookback
    min_pullback: float = 0.03,   # 3% off recent high
    max_pullback: float = 0.12,   # 12% off recent high
) -> pd.DataFrame:
    """
    Build a momentum + pullback long-only signal.

    Conditions for signal = 1 (long regime):

      1) Uptrend:
           - 50-day SMA > 200-day SMA
           - Close > 50-day SMA

      2) Strong momentum:
           - (Close / Close_n_days_ago - 1) >= +15%

      3) Pullback:
           - Price is 3% to 12% below the max close of the last 10 days

      4) Turning up:
           - Today's close > yesterday's close
    """
    close = pd.to_numeric(df["close"], errors="coerce")

    # Trend filters
    df["ma_trend_short"] = close.rolling(
        window=trend_short, min_periods=trend_short
    ).mean()
    df["ma_trend_long"] = close.rolling(
        window=trend_long, min_periods=trend_long
    ).mean()

    # Momentum: ~3-month lookback
    df["momentum"] = close / close.shift(mom_lookback) - 1.0

    # Pullback from recent high
    recent_max = close.rolling(
        window=pullback_lookback,
        min_periods=pullback_lookback
    ).max()

    pullback_pct = (recent_max - close) / recent_max
    pullback_pct = pullback_pct.fillna(0.0)
    df["pullback_pct"] = pullback_pct

    # Initialize signal to 0
    df["signal"] = 0

    # Turning-up condition: today's close > yesterday's close
    close_yesterday = close.shift(1)

    long_condition = (
        (df["ma_trend_short"] > df["ma_trend_long"]) &    # uptrend
        (close > df["ma_trend_short"]) &                  # price above 50-day
        (df["momentum"] >= min_momentum) &                # strong momentum
        (df["pullback_pct"] >= min_pullback) &
        (df["pullback_pct"] <= max_pullback) &            # 3%–12% pullback
        (close > close_yesterday)                         # price turning up
    )

    df.loc[long_condition, "signal"] = 1

    return df


# ------------------------------------------------------------
# 2. BACKTEST ENGINE USING THE RISK MODULE
# ------------------------------------------------------------

def run_backtest(
    df: pd.DataFrame,
    risk_config: RiskConfig,
    initial_equity: float = 100_000.0,
    max_hold_days: int = 10,
    trail_atr_multiple: float | None = None,
) -> tuple[pd.DataFrame, float]:
    """
    Run a simple long-only backtest:

      - Uses 'signal' column (1 = long regime, 0 = flat)
      - Enters when signal goes 0 -> 1
      - Exits on:
          * stop hit (with optional ATR trailing)
          * target hit
          * signal goes 1 -> 0
          * max_hold_days reached

      - Position size, stop, and target are all computed by calculate_risk_for_trade.

      - If trail_atr_multiple is not None, a trailing stop is applied:
          * Each bar, stop_price is raised to (close - trail_atr_multiple * ATR)
            if that is higher than the current stop_price.
          * The stop never moves down.
    """
    df = df.reset_index(drop=True).copy()

    equity = initial_equity
    in_position = False
    entry_index: int | None = None
    current_trade: dict | None = None

    trades: list[dict] = []

    # Start at 1 so we can always look back at i-1
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]

        # Ensure numeric types
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])
        signal = int(row["signal"])
        prev_signal = int(prev_row["signal"])

        if not in_position:
            # ENTRY: signal turns from 0 -> 1
            if prev_signal == 0 and signal == 1:
                entry_price = close
                atr = float(row["atr"])

                # Compute risk + position size for this trade
                risk_info = calculate_risk_for_trade(
                    entry_price=entry_price,
                    atr=atr,
                    account_equity=equity,
                    config=risk_config,
                )

                shares = int(risk_info["shares"])
                if shares <= 0:
                    # Skip trades where even 1 share would violate risk rules
                    continue

                stop_price = float(risk_info["stop_price"])
                target_price = float(risk_info["target_price"])

                in_position = True
                entry_index = i

                current_trade = {
                    "entry_date": row["date"],
                    "entry_price": entry_price,
                    "stop_price": stop_price,             # dynamic stop (may trail)
                    "initial_stop_price": stop_price,     # fixed for R calculation
                    "target_price": target_price,
                    "shares": shares,
                    "allowed_dollar_risk": float(risk_info["allowed_dollar_risk"]),
                    "account_equity_at_entry": equity,
                }

        else:
            # We are IN a position: manage exits
            assert current_trade is not None
            assert entry_index is not None

            hold_days = i - entry_index

            # ---------------- Trailing stop update (if enabled) ----------------
            if trail_atr_multiple is not None:
                atr_today = float(row.get("atr", 0.0))
                if atr_today > 0.0:
                    trail_dist = trail_atr_multiple * atr_today
                    candidate_stop = close - trail_dist
                    # For a long, stop can only move up
                    if candidate_stop > current_trade["stop_price"]:
                        current_trade["stop_price"] = candidate_stop

            exit_price: float | None = None
            exit_reason: str | None = None

            # 1) Stop-loss: if today's low touches or breaks (possibly trailed) stop
            if low <= current_trade["stop_price"]:
                exit_price = current_trade["stop_price"]
                exit_reason = "stop"

            # 2) Target: if today's high touches or exceeds target
            elif high >= current_trade["target_price"]:
                exit_price = current_trade["target_price"]
                exit_reason = "target"

            # 3) Signal exit: trend turns off (1 -> 0)
            elif prev_signal == 1 and signal == 0:
                exit_price = close
                exit_reason = "signal_exit"

            # 4) Time-based exit
            elif hold_days >= max_hold_days:
                exit_price = close
                exit_reason = "time_exit"

            if exit_price is not None:
                # Compute R and P&L (R based on INITIAL stop, not trailed stop)
                R_value = calculate_R(
                    exit_price=exit_price,
                    entry_price=current_trade["entry_price"],
                    stop_price=current_trade["initial_stop_price"],
                )

                pnl_dollars = (exit_price - current_trade["entry_price"]) * current_trade["shares"]
                equity += pnl_dollars

                current_trade.update({
                    "exit_date": row["date"],
                    "exit_price": exit_price,
                    "R": R_value,
                    "pnl_dollars": pnl_dollars,
                    "equity_after": equity,
                    "exit_reason": exit_reason,
                })

                trades.append(current_trade)

                # Reset position state
                in_position = False
                entry_index = None
                current_trade = None

    trades_df = pd.DataFrame(trades)
    return trades_df, equity


# ------------------------------------------------------------
# 3. REPORTING
# ------------------------------------------------------------

def print_summary(trades_df: pd.DataFrame, initial_equity: float, final_equity: float) -> None:
    print("\n===== BACKTEST SUMMARY =====")

    if trades_df.empty:
        print("No trades were taken.")
        print(f"Final equity: {final_equity:,.2f} (initial: {initial_equity:,.2f})")
        return

    n_trades = len(trades_df)
    wins = trades_df[trades_df["pnl_dollars"] > 0]
    losses = trades_df[trades_df["pnl_dollars"] <= 0]

    win_rate = len(wins) / n_trades * 100.0
    avg_R = trades_df["R"].mean()
    total_R = trades_df["R"].sum()

    # Equity curve (include starting equity)
    equity_series = pd.concat(
        [pd.Series([initial_equity]), trades_df["equity_after"]],
        ignore_index=True,
    )

    rolling_max = equity_series.cummax()
    drawdown = equity_series - rolling_max
    max_drawdown = drawdown.min()

    print(f"Number of trades      : {n_trades}")
    print(f"Win rate              : {win_rate:.2f}%")
    print(f"Average R per trade   : {avg_R:.2f}R")
    print(f"Total R               : {total_R:.2f}R")
    print(f"Initial equity        : {initial_equity:,.2f}")
    print(f"Final equity          : {final_equity:,.2f}")
    print(f"Net P&L               : {final_equity - initial_equity:,.2f}")
    print(f"Max drawdown          : {max_drawdown:,.2f}")


def analyze_combined_trades(combined_trades: pd.DataFrame) -> None:
    """
    Extra analysis across all symbols:
      - Per-symbol performance (R, win rate, drawdown)
      - R-distribution buckets
    """
    if combined_trades.empty:
        print("\nNo trades to analyze.")
        return

    print("\n\n===== PER-SYMBOL PERFORMANCE =====")
    grouped = combined_trades.groupby("symbol")

    for symbol, g in grouped:
        n_trades = len(g)
        wins = g[g["pnl_dollars"] > 0]
        losses = g[g["pnl_dollars"] <= 0]

        win_rate = len(wins) / n_trades * 100.0 if n_trades > 0 else 0.0
        avg_R = g["R"].mean()
        total_R = g["R"].sum()

        # Reconstruct a simple equity curve per symbol
        if "account_equity_at_entry" in g.columns:
            start_equity = float(g["account_equity_at_entry"].iloc[0])
        else:
            start_equity = 100_000.0  # fallback

        equity_series = pd.concat(
            [pd.Series([start_equity]), g["equity_after"]],
            ignore_index=True,
        )
        rolling_max = equity_series.cummax()
        drawdown = equity_series - rolling_max
        max_drawdown = drawdown.min()

        print(f"\nSymbol              : {symbol}")
        print(f"  Trades            : {n_trades}")
        print(f"  Win rate          : {win_rate:.2f}%")
        print(f"  Avg R per trade   : {avg_R:.2f}R")
        print(f"  Total R           : {total_R:.2f}R")
        print(f"  Start equity      : {start_equity:,.2f}")
        print(f"  End equity        : {float(g['equity_after'].iloc[-1]):,.2f}")
        print(f"  Max drawdown      : {max_drawdown:,.2f}")

    # ---- Global R-distribution across all symbols ----
    print("\n\n===== GLOBAL R DISTRIBUTION (ALL SYMBOLS) =====")

    R_values = combined_trades["R"].dropna()
    if R_values.empty:
        print("No R values available.")
        return

    bins = [-10.0, -1.0, 0.0, 1.0, 2.0, 10.0]
    labels = ["< -1R", "-1R to 0R", "0R to 1R", "1R to 2R", "> 2R"]

    R_binned = pd.cut(R_values, bins=bins, labels=labels, include_lowest=True)
    counts = R_binned.value_counts().reindex(labels, fill_value=0)

    total_trades = len(R_values)
    for label in labels:
        count = int(counts[label])
        pct = (count / total_trades) * 100.0 if total_trades > 0 else 0.0
        print(f"{label:10s}: {count:4d} trades ({pct:5.1f}%)")


# ------------------------------------------------------------
# 4. PER-SYMBOL BACKTEST HELPER
# ------------------------------------------------------------

def run_backtest_for_symbol(
    symbol: str,
    project_root: Path,
    risk_config: RiskConfig,
    initial_equity: float,
    max_hold_days: int,
    trail_atr_multiple: float | None,
) -> tuple[pd.DataFrame, float]:
    """
    Load data for one symbol, build indicators/signals, run backtest,
    print trades + summary, and return the trades + final equity.
    """
    data_path = project_root / "project_purple" / "data" / f"{symbol}_daily.csv"
    print(f"\nLoading data for {symbol} from: {data_path}")

    if not data_path.exists():
        raise FileNotFoundError(f"Could not find data file for {symbol} at: {data_path}")

    df_raw = pd.read_csv(data_path)
    print("Raw columns:", list(df_raw.columns))

    df = clean_aapl_dataframe(df_raw)
    print("Cleaned columns:", list(df.columns))

    df = add_atr(df)
    df = add_momentum_pullback_signals(df)

    df = df.dropna(
        subset=["atr", "ma_trend_short", "ma_trend_long", "momentum", "pullback_pct"]
    ).reset_index(drop=True)

    trades_df, final_equity = run_backtest(
        df=df,
        risk_config=risk_config,
        initial_equity=initial_equity,
        max_hold_days=max_hold_days,
        trail_atr_multiple=trail_atr_multiple,
    )

    if not trades_df.empty:
        trades_df["symbol"] = symbol
        print(f"\n===== FIRST 10 TRADES FOR {symbol} =====")
        print(trades_df.head(10))
    else:
        print(f"\nNo trades were generated by the strategy for {symbol}.")

    print_summary(trades_df, initial_equity, final_equity)

    return trades_df, final_equity


# ------------------------------------------------------------
# 5. MAIN ENTRY POINT
# ------------------------------------------------------------

def main():
    project_root = Path(__file__).resolve().parents[1]

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
    # Set to None to disable trailing and use only fixed stop/target.
    trail_atr_multiple: float | None = 3.0

    # Add or remove symbols here as you like
    symbols = ["AAPL", "NVDA", "SPY", "TSLA", "AMD", "MSFT", "META"]

    all_trades: list[pd.DataFrame] = []

    for symbol in symbols:
        print(f"\n\n========== RUNNING BACKTEST FOR {symbol} ==========")
        trades_df, final_equity = run_backtest_for_symbol(
            symbol=symbol,
            project_root=project_root,
            risk_config=risk_config,
            initial_equity=initial_equity,
            max_hold_days=max_hold_days,
            trail_atr_multiple=trail_atr_multiple,
        )
        if not trades_df.empty:
            all_trades.append(trades_df)

    if all_trades:
        combined_trades = pd.concat(all_trades, ignore_index=True)
        print("\n\n===== COMBINED TRADES (ALL SYMBOLS) – FIRST 10 ROWS =====")
        print(combined_trades.head(10))

        # Deep dive analysis across all symbols
        analyze_combined_trades(combined_trades)
    else:
        print("\nNo trades were generated for any symbol.")


if __name__ == "__main__":
    main()
