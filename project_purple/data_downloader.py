import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yfinance as yf

# Use the same candidate universe as the scanner so everything stays in sync.
# scanner_simple.py defines CANDIDATE_SYMBOLS, which is our curated list of
# liquid, swing-friendly tickers across sectors.
#
# This import needs to work whether you run:
#   python .\project_purple\data_downloader.py
# or (later) run it as a module.
try:
    from project_purple.scanner_simple import CANDIDATE_SYMBOLS  # type: ignore
except Exception:
    from scanner_simple import CANDIDATE_SYMBOLS  # type: ignore

# Backward compatibility: other parts of the project may still reference TICKERS.
TICKERS = CANDIDATE_SYMBOLS


def _repo_root() -> Path:
    # __file__ = .../project_purple/project_purple/data_downloader.py
    # parents[0] -> .../project_purple/project_purple
    # parents[1] -> .../project_purple (repo root)
    return Path(__file__).resolve().parents[1]


def _canonical_data_dir() -> Path:
    """
    Canonical location for market data is repo-root /data (Option A).
    This function makes it impossible for the downloader to recreate
    the old, dangerous package folder: project_purple/data
    """
    data_dir = _repo_root() / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def _validate_ohlcv(df: pd.DataFrame, symbol: str) -> None:
    """
    Raise ValueError if we detect obviously corrupt OHLCV.
    We keep validation minimal and focused on the known failure mode:
    high < low (and NaNs in OHLC).
    """
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{symbol}: missing columns {missing}")

    # NaNs in OHLC are not acceptable for our pipeline.
    if df[["open", "high", "low", "close"]].isna().any().any():
        raise ValueError(f"{symbol}: NaNs detected in OHLC columns")

    # Known corruption failure mode
    bad = (df["high"] < df["low"]).sum()
    if bad > 0:
        raise ValueError(f"{symbol}: Validation failed: high < low on {bad} rows")


def _build_project_csv(symbol: str, hist: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a yfinance history() DataFrame into Project Purple's expected CSV shape:

        columns: Price, symbol, open, high, low, close, volume

    Header rows:
        Row 0: Ticker,<sym>,<sym>,<sym>,<sym>,<sym>,<sym>
        Row 1: date,,,,,,

    Data rows:
        YYYY-MM-DD,<sym>,open,high,low,close,volume
    """
    # Ensure required columns exist (yfinance history usually gives these)
    # We select only what we need so dividends/splits never interfere.
    needed_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in needed_cols if c not in hist.columns]
    if missing:
        raise ValueError(f"{symbol}: yfinance history missing columns: {missing}")

    # Build body
    dates = pd.to_datetime(hist.index).strftime("%Y-%m-%d")

    body = pd.DataFrame(
        {
            "Price": dates,
            "symbol": [symbol] * len(hist),
            "open": pd.Series(hist["Open"].to_numpy().ravel()),
            "high": pd.Series(hist["High"].to_numpy().ravel()),
            "low": pd.Series(hist["Low"].to_numpy().ravel()),
            "close": pd.Series(hist["Close"].to_numpy().ravel()),
            "volume": pd.Series(hist["Volume"].to_numpy().ravel()),
        }
    )

    _validate_ohlcv(body, symbol)

    header_rows = pd.DataFrame(
        {
            "Price": ["Ticker", "date"],
            "symbol": [symbol, ""],
            "open": [symbol, ""],
            "high": [symbol, ""],
            "low": [symbol, ""],
            "close": [symbol, ""],
            "volume": [symbol, ""],
        }
    )

    final_df = pd.concat([header_rows, body], ignore_index=True)
    final_df = final_df[["Price", "symbol", "open", "high", "low", "close", "volume"]]
    return final_df


def download_and_save_daily_data(
    symbols: Optional[List[str]] = None,
    period: str = "10y",
    interval: str = "1d",
    overwrite: bool = False,
) -> None:
    """
    Download daily OHLCV data using yfinance and save CSVs to canonical repo-root /data.

    Safety features:
    - Writes ONLY to repo-root /data (never project_purple/data).
    - Uses yfinance Ticker().history() + explicit column names (Open/High/Low/Close/Volume).
    - Validates high >= low and rejects NaNs in OHLC.
    - If overwrite=False and the standard file exists, writes a timestamped "download" file
      instead of overwriting your known-good dataset.
    """
    data_dir = _canonical_data_dir()

    use_symbols = symbols if symbols else list(TICKERS)

    print(f"Saving CSVs to (canonical): {data_dir}")
    print(f"Symbols to download ({len(use_symbols)}): {use_symbols}")
    print(f"yfinance: period={period}, interval={interval}, overwrite={overwrite}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for symbol in use_symbols:
        print(f"\nDownloading {symbol} daily data with yfinance...")

        try:
            # Ticker().history() is typically more stable than yf.download() for single symbols
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
            print(
                f"  WARNING: No data returned for {symbol}. "
                f"Check ticker validity or yfinance availability."
            )
            continue

        try:
            final_df = _build_project_csv(symbol, hist)
        except Exception as e:
            print(f"  ERROR: Could not build validated CSV for {symbol}: {e}")
            continue

        standard_path = data_dir / f"{symbol}_daily.csv"

        # If we aren't overwriting, never clobber an existing "known good" dataset.
        if standard_path.exists() and not overwrite:
            out_path = data_dir / f"{symbol}_daily.download_{timestamp}.csv"
            print(
                f"  NOTE: {standard_path.name} already exists; "
                f"writing to {out_path.name} instead (use --overwrite to replace)."
            )
        else:
            out_path = standard_path

        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

        try:
            final_df.to_csv(tmp_path, index=False)
            tmp_path.replace(out_path)
            print(f"  Saved {symbol} to: {out_path}")
        except PermissionError:
            print(
                f"  ERROR: Permission denied writing {out_path}.\n"
                f"         Close Excel or any program using this file, then run again."
            )
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
        except Exception as e:
            print(f"  ERROR saving {symbol} to {out_path}: {e}")
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
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
             "If not set, existing files are preserved and a timestamped download file is written.",
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
