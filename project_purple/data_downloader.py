import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yfinance as yf

# Use the same candidate universe as the scanner so everything stays in sync.
try:
    from project_purple.scanner_simple import CANDIDATE_SYMBOLS  # type: ignore
except Exception:
    from scanner_simple import CANDIDATE_SYMBOLS  # type: ignore

# Use the SAME canonical data-dir resolver as the loader (Option A: repo-root ./data)
try:
    from project_purple.data_loader import get_data_dir  # type: ignore
except Exception:
    from data_loader import get_data_dir  # type: ignore

# Backward compatibility: other parts of the project may still reference TICKERS.
TICKERS = CANDIDATE_SYMBOLS


def _validate_ohlcv(df: pd.DataFrame, symbol: str) -> None:
    """
    Raise ValueError if we detect obviously corrupt OHLCV.
    Focused validation (matches the known failure modes):
      - NaNs in OHLC
      - high < low
    """
    required = ["date", "symbol", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{symbol}: missing columns {missing}")

    if df[["open", "high", "low", "close"]].isna().any().any():
        raise ValueError(f"{symbol}: NaNs detected in OHLC columns")

    bad = (df["high"] < df["low"]).sum()
    if bad > 0:
        raise ValueError(f"{symbol}: Validation failed: high < low on {bad} rows")


def _build_project_df(symbol: str, hist: pd.DataFrame) -> pd.DataFrame:
    """
    Convert yfinance history() dataframe to Project Purple canonical daily format:

        date, symbol, open, high, low, close, volume

    No extra "Ticker/date" header rows (those caused confusion and extra cleaning).
    """
    needed_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in needed_cols if c not in hist.columns]
    if missing:
        raise ValueError(f"{symbol}: yfinance history missing columns: {missing}")

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(hist.index).tz_localize(None),
            "symbol": symbol,
            "open": pd.to_numeric(hist["Open"].to_numpy().ravel(), errors="coerce"),
            "high": pd.to_numeric(hist["High"].to_numpy().ravel(), errors="coerce"),
            "low": pd.to_numeric(hist["Low"].to_numpy().ravel(), errors="coerce"),
            "close": pd.to_numeric(hist["Close"].to_numpy().ravel(), errors="coerce"),
            "volume": pd.to_numeric(hist["Volume"].to_numpy().ravel(), errors="coerce"),
        }
    )

    df = df.dropna(subset=["date", "open", "high", "low", "close"]).copy()
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

    _validate_ohlcv(df, symbol)
    return df


def _read_existing_clean(csv_path: Path, symbol: str) -> pd.DataFrame:
    """
    Read an existing CSV and normalize it into canonical columns.
    Handles both:
      - legacy format: Price, symbol, open, high, low, close, volume (+ junk header rows)
      - canonical format: date, symbol, open, high, low, close, volume
    """
    raw = pd.read_csv(csv_path)

    df = raw.copy()

    # legacy "Price" -> canonical "date"
    if "Price" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"Price": "date"})

    required = {"date", "symbol", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{symbol}: existing CSV missing columns: {sorted(missing)}")

    # strip junk rows like "Ticker", "date", blanks
    date_str = df["date"].astype(str).str.strip().str.lower()
    bad_tokens = {"ticker", "date", "", "nan", "none"}
    df = df[~date_str.isin(bad_tokens)].copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close"]).copy()

    df = df[["date", "symbol", "open", "high", "low", "close", "volume"]].copy()
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()

    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

    _validate_ohlcv(df, symbol)
    return df


def _atomic_write_csv(df: pd.DataFrame, out_path: Path) -> None:
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(out_path)


def download_and_save_daily_data(
    symbols: Optional[List[str]] = None,
    period: str = "10y",
    interval: str = "1d",
    overwrite: bool = False,
) -> None:
    """
    Download daily OHLCV data using yfinance and save CSVs to canonical repo-root /data.

    Behavior:
      - overwrite=True  -> replace <SYMBOL>_daily.csv
      - overwrite=False -> SAFE UPDATE:
            if file exists, merge new rows into it (dedupe by date),
            write backup, then write updated canonical CSV to <SYMBOL>_daily.csv

    This prevents the "download_*.csv exists but system still reads stale <SYMBOL>_daily.csv" trap.
    """
    data_dir = get_data_dir()

    use_symbols = symbols if symbols else list(TICKERS)

    print(f"Saving CSVs to (canonical): {data_dir}")
    print(f"Symbols to download ({len(use_symbols)}): {use_symbols}")
    print(f"yfinance: period={period}, interval={interval}, overwrite={overwrite}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for symbol in use_symbols:
        symbol = symbol.upper().strip()
        print(f"\nDownloading {symbol} daily data with yfinance...")

        try:
            hist = yf.Ticker(symbol).history(
                period=period,
                interval=interval,
                auto_adjust=False,
                actions=False,
            )
        except Exception as e:
            print(f"  ERROR: yfinance failed for {symbol}: {e}")
            continue

        if hist is None or hist.empty:
            print(f"  WARNING: No data returned for {symbol}.")
            continue

        try:
            new_df = _build_project_df(symbol, hist)
        except Exception as e:
            print(f"  ERROR: Could not build validated dataframe for {symbol}: {e}")
            continue

        standard_path = data_dir / f"{symbol}_daily.csv"

        # If overwrite is requested, just write canonical and be done.
        if overwrite or not standard_path.exists():
            try:
                _atomic_write_csv(new_df, standard_path)
                print(f"  Saved {symbol} to: {standard_path}")
                print(f"  Rows: {len(new_df)} | Last date: {new_df['date'].iloc[-1].date()}")
            except PermissionError:
                print(
                    f"  ERROR: Permission denied writing {standard_path}.\n"
                    f"         Close Excel or any program using this file, then run again."
                )
            except Exception as e:
                print(f"  ERROR saving {symbol} to {standard_path}: {e}")
            continue

        # SAFE UPDATE path: merge existing + new, backup old file, then write merged
        try:
            existing_df = _read_existing_clean(standard_path, symbol)
        except Exception as e:
            print(f"  WARNING: Could not read/clean existing {standard_path.name}: {e}")
            print("  Proceeding by writing NEW canonical file (backup created).")
            existing_df = pd.DataFrame(columns=["date", "symbol", "open", "high", "low", "close", "volume"])

        merged = pd.concat([existing_df, new_df], ignore_index=True)
        merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
        merged = merged.dropna(subset=["date"]).sort_values("date")
        merged = merged.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

        try:
            _validate_ohlcv(merged, symbol)
        except Exception as e:
            print(f"  ERROR: merged data failed validation for {symbol}: {e}")
            print("  Keeping existing file unchanged.")
            continue

        backup_path = data_dir / f"{symbol}_daily.bak_{timestamp}.csv"
        try:
            # backup existing file first
            standard_path.replace(backup_path)
            # write merged to standard path
            _atomic_write_csv(merged, standard_path)
            print(f"  Updated {symbol} safely:")
            print(f"    Backup: {backup_path.name}")
            print(f"    Current: {standard_path.name}")
            print(f"    Rows: {len(merged)} | Last date: {merged['date'].iloc[-1].date()}")
        except PermissionError:
            print(
                f"  ERROR: Permission denied updating {standard_path}.\n"
                f"         Close Excel or any program using this file, then run again."
            )
            # attempt to restore backup if we already moved it
            try:
                if backup_path.exists() and not standard_path.exists():
                    backup_path.replace(standard_path)
            except Exception:
                pass
        except Exception as e:
            print(f"  ERROR updating {symbol}: {e}")
            # attempt to restore backup
            try:
                if backup_path.exists() and not standard_path.exists():
                    backup_path.replace(standard_path)
            except Exception:
                pass


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download daily OHLCV CSVs to repo-root /data.")
    p.add_argument(
        "--symbols",
        nargs="*",
        default=None,
        help="Optional list of symbols to download (e.g., --symbols SPY AAPL). "
             "If omitted, uses scanner CANDIDATE_SYMBOLS.",
    )
    p.add_argument("--period", default="10y", help="yfinance period (default: 10y)")
    p.add_argument("--interval", default="1d", help="yfinance interval (default: 1d)")
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing <SYMBOL>_daily.csv files in /data. "
             "If not set, existing files are safely updated via merge + backup.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    download_and_save_daily_data(
        symbols=args.symbols,
        period=args.period,
        interval=args.interval,
        overwrite=args.overwrite,
    )
