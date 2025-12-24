import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Support running as:
# - script from repo root: python project_purple/backtest_v2.py  (imports "risk")
# - module import: python -c "import project_purple.backtest_v2" (imports "project_purple.risk")
try:
    from risk import RiskConfig, calculate_risk_for_trade, calculate_R
except ModuleNotFoundError:  # pragma: no cover
    from project_purple.risk import RiskConfig, calculate_risk_for_trade, calculate_R


SLIPPAGE_PCT = 0.001


# ------------------------------------------------------------
# 0. STRICT DATA VALIDATION (Step 4)
# ------------------------------------------------------------

def validate_ohlcv_dataframe(df: pd.DataFrame, symbol: str) -> None:
    """
    Strict OHLCV validation:
    - dates must be sorted ascending and unique
    - no missing OHLC values
    - prices must be positive
    - high >= low
    - open/close must be within [low, high]
    - volume must be non-negative (if present)
    """
    required = {"date", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[{symbol}] Validation failed: missing columns {sorted(missing)}")

    if df.empty:
        raise ValueError(f"[{symbol}] Validation failed: dataframe is empty")

    if df["date"].isna().any():
        raise ValueError(f"[{symbol}] Validation failed: date contains NaT/NaN")

    # sorted + unique dates
    if not df["date"].is_monotonic_increasing:
        raise ValueError(f"[{symbol}] Validation failed: dates not sorted ascending")

    dupes = df["date"].duplicated()
    if dupes.any():
        examples = df.loc[dupes, ["date"]].head(10).to_dict(orient="records")
        raise ValueError(f"[{symbol}] Validation failed: duplicate dates (examples: {examples})")

    # numeric + finite + positive
    for c in ["open", "high", "low", "close"]:
        if df[c].isna().any():
            raise ValueError(f"[{symbol}] Validation failed: {c} has NaN")
        if not np.isfinite(df[c]).all():
            raise ValueError(f"[{symbol}] Validation failed: {c} has non-finite values")
        if (df[c] <= 0).any():
            bad = df.loc[df[c] <= 0, ["date", "open", "high", "low", "close"]].head(10).to_dict(orient="records")
            raise ValueError(f"[{symbol}] Validation failed: {c} <= 0 (examples: {bad})")

    # OHLC relationships
    bad_hl = df["high"] < df["low"]
    if bad_hl.any():
        bad = df.loc[bad_hl, ["date", "open", "high", "low", "close"]].head(10).to_dict(orient="records")
        raise ValueError(f"[{symbol}] Validation failed: high < low (examples: {bad})")

    bad_open = (df["open"] < df["low"]) | (df["open"] > df["high"])
    if bad_open.any():
        bad = df.loc[bad_open, ["date", "open", "high", "low", "close"]].head(10).to_dict(orient="records")
        raise ValueError(f"[{symbol}] Validation failed: open outside [low, high] (examples: {bad})")

    bad_close = (df["close"] < df["low"]) | (df["close"] > df["high"])
    if bad_close.any():
        bad = df.loc[bad_close, ["date", "open", "high", "low", "close"]].head(10).to_dict(orient="records")
        raise ValueError(f"[{symbol}] Validation failed: close outside [low, high] (examples: {bad})")

    # volume (if present)
    if "volume" in df.columns:
        vol = pd.to_numeric(df["volume"], errors="coerce")
        if vol.isna().any():
            raise ValueError(f"[{symbol}] Validation failed: volume has NaN/non-numeric")
        if (vol < 0).any():
            bad = df.loc[vol < 0, ["date", "volume"]].head(10).to_dict(orient="records")
            raise ValueError(f"[{symbol}] Validation failed: volume < 0 (examples: {bad})")


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
    """
    Fill realism (EOD system):
    - Signals computed on close.
    - If signal flips 0→1 on day i close, entry executes on day i+1 open.
    - Stops/targets are intraday (gap at open handled explicitly).
    - Discretionary exits (signal/time) are decided on the close and execute next open.
    """
    df = df.reset_index(drop=True).copy()
    equity = float(initial_equity)

    in_position = False
    entry_index: int | None = None
    current_trade: dict | None = None
    trades: list[dict] = []

    pending_entry: dict | None = None
    pending_exit_reason: str | None = None  # exit at next open, unless stop/target triggers first

    n = len(df)

    for i in range(1, n):
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]

        open_ = float(row["open"])
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])

        signal = int(row["signal"])
        prev_signal = int(prev_row["signal"])

        # 2A) Execute pending entry at today's open
        if (not in_position) and (pending_entry is not None):
            entry_price = open_ * (1.0 + SLIPPAGE_PCT)

            atr = float(pending_entry["atr"])
            ma_fast = float(pending_entry["ma_fast"])
            ma_slow = float(pending_entry["ma_slow"])
            decision_date = pending_entry["decision_date"]
            decision_close = float(pending_entry["decision_close"])
            entry_score = float(pending_entry["entry_score"])

            risk_info = calculate_risk_for_trade(
                entry_price=entry_price,
                atr=atr,
                account_equity=equity,
                config=risk_config,
            )

            shares = int(risk_info["shares"])
            if shares > 0:
                stop_price = float(risk_info["stop_price"])
                target_price = float(risk_info["target_price"])

                in_position = True
                entry_index = i
                current_trade = {
                    "signal_date": decision_date,
                    "signal_close": decision_close,
                    "entry_date": row["date"],
                    "entry_open": open_,
                    "entry_price": entry_price,
                    "stop_price": stop_price,
                    "initial_stop_price": stop_price,
                    "target_price": target_price,
                    "shares": shares,
                    "allowed_dollar_risk": float(risk_info["allowed_dollar_risk"]),
                    "account_equity_at_entry": equity,
                    "entry_score": entry_score,
                    "entry_atr": atr,
                    "entry_ma_fast": ma_fast,
                    "entry_ma_slow": ma_slow,
                }

            pending_entry = None

        # 2B) Exits (gap -> intraday -> scheduled exit at open)
        if in_position:
            assert current_trade is not None
            assert entry_index is not None

            hold_days = i - entry_index
            stop_price = float(current_trade["stop_price"])
            target_price = float(current_trade["target_price"])

            exit_price: float | None = None
            exit_reason: str | None = None

            if open_ <= stop_price:
                exit_price = open_
                exit_reason = "stop_gap"
            elif open_ >= target_price:
                exit_price = open_
                exit_reason = "target_gap"
            else:
                if low <= stop_price:
                    exit_price = stop_price
                    exit_reason = "stop"
                elif high >= target_price:
                    exit_price = target_price
                    exit_reason = "target"
                elif pending_exit_reason is not None:
                    exit_price = open_
                    exit_reason = pending_exit_reason

            if exit_price is not None:
                exit_price = float(exit_price) * (1.0 - SLIPPAGE_PCT)

                R_value = calculate_R(
                    exit_price=exit_price,
                    entry_price=float(current_trade["entry_price"]),
                    stop_price=float(current_trade["initial_stop_price"]),
                )

                pnl_dollars = (exit_price - float(current_trade["entry_price"])) * int(current_trade["shares"])
                equity += pnl_dollars

                current_trade.update({
                    "exit_date": row["date"],
                    "exit_open": open_,
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
                pending_exit_reason = None

            else:
                # schedule close-decided exits for next open
                if i < (n - 1):
                    if (prev_signal == 1) and (signal == 0):
                        pending_exit_reason = "signal_exit"
                    elif hold_days >= max_hold_days:
                        pending_exit_reason = "time_exit"

                # trailing stop AFTER exit checks
                if (trail_atr_multiple is not None) and (current_trade is not None):
                    atr_today = float(row.get("atr", 0.0))
                    if atr_today > 0.0:
                        candidate_stop = close - (trail_atr_multiple * atr_today)
                        if candidate_stop > float(current_trade["stop_price"]):
                            current_trade["stop_price"] = candidate_stop

        # 2C) Schedule entry for next open
        if (not in_position) and (pending_entry is None):
            if (prev_signal == 0) and (signal == 1) and (i < (n - 1)):
                atr = float(row["atr"])
                ma_fast = float(row.get("ma_fast", np.nan))
                ma_slow = float(row.get("ma_slow", np.nan))
                entry_score = _compute_entry_score(close=close, ma_fast=ma_fast, ma_slow=ma_slow, atr=atr)

                pending_entry = {
                    "decision_date": row["date"],
                    "decision_close": close,
                    "atr": atr,
                    "ma_fast": ma_fast,
                    "ma_slow": ma_slow,
                    "entry_score": entry_score,
                }

    # 2D) Force close at final close (bookkeeping convenience)
    if in_position and (current_trade is not None):
        last_row = df.iloc[-1]
        final_close = float(last_row["close"])
        exit_price = final_close * (1.0 - SLIPPAGE_PCT)

        R_value = calculate_R(
            exit_price=exit_price,
            entry_price=float(current_trade["entry_price"]),
            stop_price=float(current_trade["initial_stop_price"]),
        )

        pnl_dollars = (exit_price - float(current_trade["entry_price"])) * int(current_trade["shares"])
        equity += pnl_dollars

        current_trade.update({
            "exit_date": last_row["date"],
            "exit_open": float(last_row["open"]),
            "exit_price": exit_price,
            "R": R_value,
            "pnl_dollars": pnl_dollars,
            "equity_after": equity,
            "exit_reason": "forced_final_close",
        })
        trades.append(current_trade)

    return pd.DataFrame(trades), equity


# ------------------------------------------------------------
# 3. REPORTING
# ------------------------------------------------------------

def print_summary(trades_df: pd.DataFrame, initial_equity: float, final_equity: float) -> None:
    print("\n===== BACKTEST SUMMARY =====")
    if trades_df.empty:
        print("No trades generated.")
        print(f"Initial equity        : {initial_equity:,.2f}")
        print(f"Final equity          : {final_equity:,.2f}")
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
    """
    Resolution order:
    1) explicit data_dir argument
    2) explicit project_root argument (project_root/data or project_root/project_purple/data)
    3) inferred repo-root data: <repo_root>/data  (preferred)
    4) fallback: <this_file_dir>/data  (project_purple/data)
    """
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

    # Inferred repo root: repo_root/project_purple/backtest_v2.py -> repo_root = parents[1]
    inferred_root = Path(__file__).resolve().parents[1]
    cand_repo_data = inferred_root / "data"
    if cand_repo_data.exists():
        return cand_repo_data

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

    try:
        df = pd.read_csv(csv_path)
        print(f"Raw columns: {list(df.columns)}")

        df = clean_aapl_dataframe(df)
        print(f"Cleaned columns: {list(df.columns)}")

        # validate immediately after clean
        validate_ohlcv_dataframe(df=df, symbol=symbol)

    except Exception as e:
        print(f"\n[DATA INVALID] Skipping {symbol}. Reason: {e}")
        return pd.DataFrame(), float(initial_equity)

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
    print(f"\n[BACKTEST_V2 DONE] symbol={symbol} trades={len(trades_df)} final_equity={final_equity:,.2f}")

    return trades_df, final_equity


def main() -> None:
    risk_config = RiskConfig()
    sym = "AAPL"

    trades_df, final_equity = run_backtest_for_symbol(
        symbol=sym,
        data_dir=None,
        risk_config=risk_config,
        initial_equity=100_000.0,
        max_hold_days=10,
        trail_atr_multiple=None,
        project_root=None,
    )
    _ = trades_df
    _ = final_equity


if __name__ == "__main__":
    main()
