from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype


# Keep this list in sync with the project’s “known symbols” expectations.
# (Downloader uses scanner symbols; loader can still keep a stable subset.)
TICKERS = [
    "AAPL",
    "NVDA",
    "SPY",
    "TSLA",
    "AMD",
    "MSFT",
    "META",
    "AMZN",
    "IBKR",
    "AMDL",
]


def get_data_dir() -> Path:
    """
    Resolve the ONE canonical data directory.

    Canonical (Option A):
      - repo-root ./data

    Safety:
      - If repo-root ./project_purple/data exists, hard-fail.
        That folder previously caused corrupted OHLCV + path ambiguity.
    """
    repo_root = Path(__file__).resolve().parents[1]

    canonical = repo_root / "data"
    legacy_bad = repo_root / "project_purple" / "data"  # DO NOT USE

    # Hard gate: this folder must not exist (prevents silent reintroduction).
    if legacy_bad.exists():
        raise RuntimeError(
            "Unsafe data directory detected:\n"
            f"  {legacy_bad}\n\n"
            "Project Purple is locked to ONE canonical data folder:\n"
            f"  {canonical}\n\n"
            "Fix:\n"
            "  1) Delete the legacy folder:\n"
            f"     Remove-Item -Recurse -Force {legacy_bad}\n"
            "  2) Re-run the system.\n"
        )

    canonical.mkdir(parents=True, exist_ok=True)
    return canonical


def _normalize_dates_to_naive_midnight(series: pd.Series, symbol: str) -> pd.Series:
    """
    Convert a date-like series that may contain:
      - 'YYYY-MM-DD'
      - 'YYYY-MM-DD 00:00:00-05:00'
      - mixed tz-aware / tz-naive strings
    into a single canonical dtype:
      datetime64[ns] (tz-naive), normalized to midnight.

    Key: utc=True makes pandas parse mixed timezones into a consistent tz-aware type,
    which we then strip back to tz-naive.
    """
    # Stringify to reduce weird mixed types (Timestamp, datetime, str, etc.)
    s = series.astype(str).str.strip()

    # Parse with utc=True to avoid "mixed timezone" object dtype.
    dt = pd.to_datetime(s, errors="coerce", utc=True)

    # Strip timezone to tz-naive datetime64[ns]
    # (tz_convert(None) removes tz info while preserving absolute time)
    dt = dt.dt.tz_convert(None)

    # Normalize to midnight (date-only semantics)
    dt = dt.dt.normalize()

    return dt


def _clean_daily_csv(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Normalize your CSVs into:
      date, symbol, open, high, low, close, volume

    Handles:
      - the "Price" date column
      - junk rows like: Ticker,<sym>,<sym>,... or date,,,,,,
      - mixed tz-aware/tz-naive date strings
    """
    df = raw.copy()

    # Standardize date column name
    if "Price" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"Price": "date"})

    # If symbol column is missing, create it (we'll force it anyway later)
    if "symbol" not in df.columns:
        df["symbol"] = symbol

    required = {"date", "symbol", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {symbol} CSV: {sorted(missing)}")

    # Remove obvious junk header rows that appear inside the file
    date_str = df["date"].astype(str).str.strip().str.lower()
    bad_tokens = {"ticker", "date", "price", "", "nan", "none"}
    df = df[~date_str.isin(bad_tokens)].copy()

    # Canonicalize dates: tz-safe, tz-naive, midnight-normalized
    df["date"] = _normalize_dates_to_naive_midnight(df["date"], symbol=symbol)
    df = df.dropna(subset=["date"]).copy()

    # Force symbol to match the filename / requested symbol (single source of truth)
    df["symbol"] = symbol.upper().strip()

    # Numeric conversion
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with missing OHLCV
    df = df.dropna(subset=["open", "high", "low", "close", "volume"]).copy()

    # Ensure expected column order
    expected_cols = ["date", "symbol", "open", "high", "low", "close", "volume"]
    df = df.reindex(columns=expected_cols)

    # Sort + de-dup dates (keep last)
    df = (
        df.sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )

    return df


def _validate_ohlcv_dataframe(df: pd.DataFrame, symbol: str) -> None:
    """
    Strict OHLCV validation for downstream consumers.
    Ensures we do not silently operate on corrupted data.
    """
    required = {"date", "symbol", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[{symbol}] Validation failed: missing columns {sorted(missing)}")

    if df.empty:
        raise ValueError(f"[{symbol}] Validation failed: dataframe is empty")

    # Date must be real datetime64 (tz-naive) and normalized to midnight
    if not is_datetime64_any_dtype(df["date"]):
        raise ValueError(f"[{symbol}] Validation failed: date is not datetime64 dtype")

    if df["date"].isna().any():
        raise ValueError(f"[{symbol}] Validation failed: date contains NaT/NaN")

    # Enforce our date-only semantics: date == normalize(date)
    if (df["date"] != df["date"].dt.normalize()).any():
        raise ValueError(f"[{symbol}] Validation failed: date not normalized to midnight")

    if not df["date"].is_monotonic_increasing:
        raise ValueError(f"[{symbol}] Validation failed: dates not sorted ascending")

    dupes = df["date"].duplicated()
    if dupes.any():
        examples = df.loc[dupes, ["date"]].head(10).to_dict(orient="records")
        raise ValueError(f"[{symbol}] Validation failed: duplicate dates (examples: {examples})")

    # Symbol must be single-valued and match the requested symbol
    if df["symbol"].nunique() != 1:
        raise ValueError(f"[{symbol}] Validation failed: symbol column has {df['symbol'].nunique()} unique values")
    if str(df["symbol"].iloc[0]).upper() != symbol.upper():
        raise ValueError(f"[{symbol}] Validation failed: symbol mismatch: col={df['symbol'].iloc[0]}")

    # OHLC must be numeric, finite, and > 0
    for c in ["open", "high", "low", "close"]:
        if df[c].isna().any():
            raise ValueError(f"[{symbol}] Validation failed: {c} has NaN")

        # NEW (Codex hardening): reject inf/-inf and other non-finite values
        if not np.isfinite(df[c].to_numpy()).all():
            raise ValueError(f"[{symbol}] Validation failed: {c} has non-finite values (inf/-inf)")

        if (df[c] <= 0).any():
            bad = df.loc[df[c] <= 0, ["date", "open", "high", "low", "close"]].head(10).to_dict(orient="records")
            raise ValueError(f"[{symbol}] Validation failed: {c} <= 0 (examples: {bad})")

    # high >= low
    bad_hl = df["high"] < df["low"]
    if bad_hl.any():
        bad = df.loc[bad_hl, ["date", "open", "high", "low", "close"]].head(10).to_dict(orient="records")
        raise ValueError(f"[{symbol}] Validation failed: high < low (examples: {bad})")

    # open/close must be within [low, high]
    bad_open = (df["open"] < df["low"]) | (df["open"] > df["high"])
    if bad_open.any():
        bad = df.loc[bad_open, ["date", "open", "high", "low", "close"]].head(10).to_dict(orient="records")
        raise ValueError(f"[{symbol}] Validation failed: open outside [low, high] (examples: {bad})")

    bad_close = (df["close"] < df["low"]) | (df["close"] > df["high"])
    if bad_close.any():
        bad = df.loc[bad_close, ["date", "open", "high", "low", "close"]].head(10).to_dict(orient="records")
        raise ValueError(f"[{symbol}] Validation failed: close outside [low, high] (examples: {bad})")

    # Volume must be numeric, finite, and >= 0
    vol = pd.to_numeric(df["volume"], errors="coerce")
    if vol.isna().any():
        raise ValueError(f"[{symbol}] Validation failed: volume has NaN/non-numeric")

    # NEW (Codex hardening): reject inf/-inf and other non-finite values
    if not np.isfinite(vol.to_numpy()).all():
        raise ValueError(f"[{symbol}] Validation failed: volume has non-finite values (inf/-inf)")

    if (vol < 0).any():
        bad = df.loc[vol < 0, ["date", "volume"]].head(10).to_dict(orient="records")
        raise ValueError(f"[{symbol}] Validation failed: volume < 0 (examples: {bad})")


def load_symbol_daily(symbol: str) -> pd.DataFrame:
    """
    Load a single symbol's daily data from CSV.

    Canonical path only:
      - repo-root ./data/<SYMBOL>_daily.csv
    """
    symbol = symbol.upper().strip()

    data_dir = get_data_dir()
    csv_path = data_dir / f"{symbol}_daily.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV for {symbol} not found at: {csv_path}\n"
            f"Canonical data dir is: {data_dir}\n"
            "If you need to fetch data, run:\n"
            f"  python -u .\\project_purple\\data_downloader.py --symbols {symbol}\n"
        )

    raw = pd.read_csv(csv_path)
    df = _clean_daily_csv(raw=raw, symbol=symbol)
    _validate_ohlcv_dataframe(df=df, symbol=symbol)
    return df


def load_all_daily() -> Dict[str, pd.DataFrame]:
    """
    Load daily data for all symbols in TICKERS into a dict:
        { "AAPL": DataFrame, "NVDA": DataFrame, ... }
    """
    data: Dict[str, pd.DataFrame] = {}

    for symbol in TICKERS:
        try:
            df = load_symbol_daily(symbol)
            data[symbol] = df
        except Exception as e:
            print(f"WARNING: Failed to load {symbol}: {e}")

    return data


def validate_all_csvs_in_data_dir() -> List[Tuple[str, str]]:
    """
    Validate *every* *_daily.csv in the canonical ./data directory.
    Returns a list of failures as (symbol, error_string). Empty list == all good.
    """
    data_dir = get_data_dir()
    paths = sorted(data_dir.glob("*_daily.csv"))
    print(f"CSV count: {len(paths)}")

    failures: List[Tuple[str, str]] = []

    for p in paths:
        symbol = p.name.replace("_daily.csv", "").upper().strip()
        try:
            raw = pd.read_csv(p)
            df = _clean_daily_csv(raw=raw, symbol=symbol)
            _validate_ohlcv_dataframe(df=df, symbol=symbol)

            last = df["date"].max()
            last_str = last.strftime("%Y-%m-%d") if pd.notna(last) else "NA"
            print(f"OK   {symbol:6s} rows={len(df):5d} last={last_str}")
        except Exception as e:
            msg = str(e).splitlines()[0]
            failures.append((symbol, msg))
            print(f"FAIL {symbol:6s} {msg}")

    print(f"\nFailures: {len(failures)}")
    for sym, msg in failures[:50]:
        print(f" - {sym}: {msg}")

    return failures


if __name__ == "__main__":
    # Simple diagnostics when you run this file directly
    print("Testing data_loader...")

    try:
        resolved = get_data_dir()
        print("Resolved data dir:", resolved)

        # HARD GATE: validate everything we have on disk
        failures = validate_all_csvs_in_data_dir()
        if failures:
            raise SystemExit(1)

        # Sanity: show SPY
        spy_df = load_symbol_daily("SPY")
        print("\nSPY head:")
        print(spy_df.head())
        print(f"\nSPY rows: {len(spy_df)}")
        print("\nSPY tail:")
        print(spy_df.tail(3)[["date", "close"]])

    except SystemExit as e:
        raise
    except Exception as e:
        print(f"Error while loading data: {e}")
        raise
