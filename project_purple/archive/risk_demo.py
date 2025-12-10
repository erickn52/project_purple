import pandas as pd
from pathlib import Path

from risk import RiskConfig, calculate_risk_for_trade, calculate_R


def clean_aapl_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the AAPL_daily.csv format you showed in Excel.

    Original structure:
      Row 1: Price | symbol | open | high | low | close | volume  (headers)
      Row 2: Ticker | AAPL | AAPL | AAPL | AAPL | AAPL | AAPL    (metadata)
      Row 3: date | ...                                          (metadata)
      Row 4+: actual data

    Steps:
      - Keep row 1 as header (pandas does this by default).
      - Drop rows where first column is 'Ticker' or 'date'.
      - Rename 'Price' column to 'date'.
      - Lowercase all column names.
    """
    # Drop metadata rows
    first_col_name = df.columns[0]

    df = df[~df[first_col_name].isin(["Ticker", "date"])].reset_index(drop=True)

    # Rename 'Price' -> 'date' if present
    rename_map = {}
    for col in df.columns:
        if col.strip().lower() == "price":
            rename_map[col] = "date"

    if rename_map:
        df = df.rename(columns=rename_map)

    # Standardize column names to lowercase with no spaces
    df.columns = [c.strip().lower() for c in df.columns]

    # At this point we expect: date, symbol, open, high, low, close, volume
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


def main():
    # 1. Locate AAPL_daily.csv inside project_purple/data
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "project_purple" / "data" / "AAPL_daily.csv"

    print(f"Loading data from: {data_path}")

    if not data_path.exists():
        raise FileNotFoundError(f"Could not find data file at: {data_path}")

    # 2. Load raw CSV (use row 1 as header)
    df_raw = pd.read_csv(data_path)

    print("\nRaw columns loaded:", list(df_raw.columns))

    # 3. Clean according to your file's structure
    df = clean_aapl_dataframe(df_raw)
    print("Cleaned columns:", list(df.columns))

    # 4. Add ATR
    df = add_atr(df)
    df = df.dropna(subset=["atr"]).reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "After computing ATR and dropping NaNs, the DataFrame is empty. "
            "Check cleaning logic or ATR period."
        )

    # 5. Take the last bar as a pretend entry
    last_row = df.iloc[-1]
    entry_price = float(last_row["close"])
    atr = float(last_row["atr"])

    account_equity = 100_000.0

    config = RiskConfig(
        risk_per_trade_pct=0.01,         # risk 1% of account per trade
        atr_stop_multiple=1.5,           # stop is 1.5 ATR below entry
        atr_target_multiple=3.0,         # target is 3 ATR above entry
        max_position_pct_of_equity=0.20, # max 20% of equity per position
        min_shares=1,
    )

    risk_info = calculate_risk_for_trade(
        entry_price=entry_price,
        atr=atr,
        account_equity=account_equity,
        config=config,
    )

    print("\n=== RISK SETUP FOR EXAMPLE AAPL TRADE ===")
    for k, v in risk_info.items():
        print(f"{k:20s}: {v}")

    # 6. Calculate R-multiple at target
    exit_price = risk_info["target_price"]
    R = calculate_R(
        exit_price=exit_price,
        entry_price=risk_info["entry_price"],
        stop_price=risk_info["stop_price"],
    )
    print(f"\nIf we exit at target: R = {R:.2f}R")


if __name__ == "__main__":
    main()
