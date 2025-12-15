import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from risk import RiskConfig, calculate_risk_for_trade, calculate_R


SLIPPAGE_PCT = 0.001


# ------------------------------------------------------------
# 1. DATA CLEANING & INDICATORS
# ------------------------------------------------------------

def _parse_date_series(date_series: pd.Series) -> pd.Series:
    s = date_series.copy()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(s, format="mixed", errors="coerce")
    except TypeError:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(s, errors="coerce")

    if parsed.isna().mean() > 0.30:
        serial = pd.to_numeric(s, errors="coerce")
        mask = serial.notna()
        if mask.any():
            serial_parsed = pd.to_datetime(
                serial[mask],
                unit="D",
                origin="1899-12-30",
                errors="coerce",
            )
            parsed.loc[mask] = serial_parsed

    return parsed


def clean_aapl_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Price" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"Price": "date"})

    required = {"date", "symbol", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

    date_str = df["date"].astype(str).str.strip()
    bad_tokens = {"ticker", "date", "", "nan", "none"}
    df = df[~date_str.str.lower().isin(bad_tokens)].copy()

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["date"] = _parse_date_series(df["date"])
    df = df.dropna(subset=["date", "open", "high", "low", "close"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df = df.copy()
    prev_close = df["close"].shift(1)

    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr"] = tr.rolling(period).mean()
    return df


def add_momentum_pullback_signals(
    df: pd.DataFrame,
    ma_fast: int = 20,
    ma_slow: int = 50,
) -> pd.DataFrame:
    df = df.copy()
    df["ma_fast"] = df["close"].rolling(ma_fast).mean()
    df["ma_slow"] = df["close"].rolling(ma_slow).mean()

    df["signal"] = 0
    cond = (df["close"] > df["ma_slow"]) & (df["ma_fast"] > df["ma_slow"])
    df.loc[cond, "signal"] = 1

    return df


def _compute_entry_score(close: float, ma_fast: float, ma_slow: float, atr: float) -> float:
    if not np.isfinite(close) or close <= 0:
        return float("-inf")
    if not np.isfinite(ma_slow) or ma_slow <= 0:
        return float("-inf")
    if not np.isfinite(ma_fast) or ma_fast <= 0:
        return float("-inf")
    if not np.isfinite(atr) or atr <= 0:
        return float("-inf")

    trend1 = (close / ma_slow) - 1.0
    trend2 = (ma_fast / ma_slow) - 1.0
    vol_penalty = (atr / close)

    score = (1.0 * trend1) + (0.5 * trend2) - (0.5 * vol_penalty)
    return float(score)


# ------------------------------------------------------------
# 2. BACKTEST CORE
# ------------------------------------------------------------

def run_backtest(
    df: pd.DataFrame,
    risk_config: RiskConfig,
    initial_equity: float = 100_000.0,
    max_hold_days: int = 10,
    trail_atr_multiple: float | None = None,
) -> tuple[pd.DataFrame, float]:
    df = df.reset_index(drop=True).copy()

    equity = float(initial_equity)
    in_position = False
    entry_index: int | None = None
    current_trade: dict | None = None
    trades: list[dict] = []

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]

        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])
        signal = int(row["signal"])
        prev_signal = int(prev_row["signal"])

        if not in_position:
            if prev_signal == 0 and signal == 1:
                entry_price = close * (1.0 + SLIPPAGE_PCT)
                atr = float(row["atr"])
                ma_fast = float(row.get("ma_fast", np.nan))
                ma_slow = float(row.get("ma_slow", np.nan))

                risk_info = calculate_risk_for_trade(
                    entry_price=entry_price,
                    atr=atr,
                    account_equity=equity,
                    config=risk_config,
                )

                shares = int(risk_info["shares"])
                if shares <= 0:
                    continue

                stop_price = float(risk_info["stop_price"])
                target_price = float(risk_info["target_price"])

                entry_score = _compute_entry_score(close=close, ma_fast=ma_fast, ma_slow=ma_slow, atr=atr)

                in_position = True
                entry_index = i
                current_trade = {
                    "entry_date": row["date"],
                    "entry_price": entry_price,
                    "stop_price": stop_price,
                    "initial_stop_price": stop_price,
                    "target_price": target_price,
                    "shares": shares,
                    "allowed_dollar_risk": float(risk_info["allowed_dollar_risk"]),
                    "account_equity_at_entry": equity,

                    # new: stored for portfolio tie-break + debugging
                    "entry_score": entry_score,
                    "entry_close": close,
                    "entry_atr": atr,
                    "entry_ma_fast": ma_fast,
                    "entry_ma_slow": ma_slow,
                }

        else:
            assert current_trade is not None
            assert entry_index is not None

            hold_days = i - entry_index

            if trail_atr_multiple is not None:
                atr_today = float(row.get("atr", 0.0))
                if atr_today > 0.0:
                    trail_dist = trail_atr_multiple * atr_today
                    candidate_stop = close - trail_dist
                    if candidate_stop > current_trade["stop_price"]:
                        current_trade["stop_price"] = candidate_stop

            exit_price: float | None = None
            exit_reason: str | None = None

            if low <= current_trade["stop_price"]:
                exit_price = current_trade["stop_price"]
                exit_reason = "stop"
            elif high >= current_trade["target_price"]:
                exit_price = current_trade["target_price"]
                exit_reason = "target"
            elif prev_signal == 1 and signal == 0:
                exit_price = close
                exit_reason = "signal_exit"
            elif hold_days >= max_hold_days:
                exit_price = close
                exit_reason = "time_exit"

            if exit_price is not None:
                exit_price = float(exit_price) * (1.0 - SLIPPAGE_PCT)

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

                in_position = False
                entry_index = None
                current_trade = None

    return pd.DataFrame(trades), equity


# ------------------------------------------------------------
# 3. REPORTING
# ------------------------------------------------------------

def print_summary(trades_df: pd.DataFrame, initial_equity: float, final_equity: float) -> None:
    print("\n===== BACKTEST SUMMARY =====")
    if trades_df.empty:
        print("No trades generated.")
        return

    n_trades = len(trades_df)
    win_rate = (trades_df["pnl_dollars"] > 0).mean() * 100.0
    avg_R = trades_df["R"].mean()
    total_R = trades_df["R"].sum()

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
    print(f"Net P&L               : {(final_equity - initial_equity):,.2f}")
    print(f"Max drawdown          : {max_drawdown:,.2f}")


def analyze_combined_trades(combined_trades: pd.DataFrame) -> None:
    if combined_trades.empty:
        print("\nNo trades to analyze.")
        return

    print("\n\n===== PER-SYMBOL PERFORMANCE =====")
    grouped = combined_trades.groupby("symbol")

    for symbol, g in grouped:
        n_trades = len(g)
        wins = g[g["pnl_dollars"] > 0]
        win_rate = (len(wins) / n_trades) * 100.0 if n_trades > 0 else 0.0
        avg_R = float(g["R"].mean()) if n_trades > 0 else 0.0
        total_R = float(g["R"].sum()) if n_trades > 0 else 0.0

        start_equity = float(g["account_equity_at_entry"].iloc[0]) if "account_equity_at_entry" in g.columns else 100_000.0

        equity_series = pd.concat(
            [pd.Series([start_equity]), g["equity_after"]],
            ignore_index=True,
        )
        rolling_max = equity_series.cummax()
        drawdown = equity_series - rolling_max
        max_drawdown = drawdown.min()

        end_equity = float(g["equity_after"].iloc[-1]) if n_trades > 0 else start_equity

        print(f"\nSymbol              : {symbol}")
        print(f"  Trades            : {n_trades}")
        print(f"  Win rate          : {win_rate:.2f}%")
        print(f"  Avg R per trade   : {avg_R:.2f}R")
        print(f"  Total R           : {total_R:.2f}R")
        print(f"  Start equity      : {start_equity:,.2f}")
        print(f"  End equity        : {end_equity:,.2f}")
        print(f"  Max drawdown      : {max_drawdown:,.2f}")

    print("\n\n===== GLOBAL R DISTRIBUTION (ALL SYMBOLS) =====")
    R_values = combined_trades["R"].dropna()
    if R_values.empty:
        print("No R values found.")
        return

    bins = [-float("inf"), -1.0, 0.0, 1.0, 2.0, float("inf")]
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

def _resolve_data_dir(data_dir: Path | None, project_root: Path | None) -> Path:
    if data_dir is not None:
        return Path(data_dir)

    if project_root is not None:
        pr = Path(project_root)
        cand1 = pr / "data"
        if cand1.exists():
            return cand1
        cand2 = pr / "project_purple" / "data"
        if cand2.exists():
            return cand2

    return Path(__file__).resolve().parent / "data"


def run_backtest_for_symbol(
    symbol: str,
    data_dir: Path | None = None,
    risk_config: RiskConfig | None = None,
    initial_equity: float = 100_000.0,
    max_hold_days: int = 10,
    trail_atr_multiple: float | None = None,
    project_root: Path | None = None,
) -> tuple[pd.DataFrame, float]:
    if risk_config is None:
        risk_config = RiskConfig()

    resolved_data_dir = _resolve_data_dir(data_dir=data_dir, project_root=project_root)
    csv_path = resolved_data_dir / f"{symbol}_daily.csv"

    if not csv_path.exists():
        print(f"Data file not found for {symbol}: {csv_path}")
        return pd.DataFrame(), float(initial_equity)

    print(f"\nLoading data for {symbol} from: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Raw columns: {list(df.columns)}")

    df = clean_aapl_dataframe(df)
    print(f"Cleaned columns: {list(df.columns)}")

    df = add_atr(df, period=14)
    df = add_momentum_pullback_signals(df, ma_fast=20, ma_slow=50)

    df = df.dropna(subset=["atr", "ma_fast", "ma_slow"]).reset_index(drop=True)
    if df.empty:
        print("No usable rows after indicators (likely too little data).")
        return pd.DataFrame(), float(initial_equity)

    trades_df, final_equity = run_backtest(
        df=df,
        risk_config=risk_config,
        initial_equity=initial_equity,
        max_hold_days=max_hold_days,
        trail_atr_multiple=trail_atr_multiple,
    )

    if not trades_df.empty:
        trades_df["symbol"] = symbol

    print_summary(trades_df, initial_equity=initial_equity, final_equity=final_equity)

    return trades_df, final_equity


def main() -> None:
    data_dir = Path(__file__).resolve().parent / "data"
    risk_config = RiskConfig()

    sym = "AAPL"
    trades_df, final_equity = run_backtest_for_symbol(
        symbol=sym,
        data_dir=data_dir,
        risk_config=risk_config,
        initial_equity=100_000.0,
        max_hold_days=10,
        trail_atr_multiple=None,
    )
    _ = trades_df
    _ = final_equity


if __name__ == "__main__":
    main()
