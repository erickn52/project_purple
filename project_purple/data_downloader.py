import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yfinance as yf

# Single source of truth for default tickers (P1 config centralization)
try:
    from project_purple.config import strategy_config  # type: ignore
except Exception:  # pragma: no cover
    # Allows running as a script from inside the package folder in some environments
    from config import strategy_config  # type: ignore


TICKERS = strategy_config.candidate_symbols


def _repo_root() -> Path:
    # __file__ = .../project_purple/project_purple/data_downloader.py
    # parents[0] -> .../project_purple/project_purple
    # parents[1] -> .../project_purple (repo root)
    return Path(__file__).resolve().parents[1]


def _canonical_data_dir() -> Path:
    """
    Canonical location for market data is repo-root /data (Option A).
    """
    data_dir = _repo_root() / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def _validate_ohlcv(df: pd.DataFrame, symbol: str) -> None:
    """
    Strict-but-practical OHLCV validation to prevent corrupted data
    from entering the pipeline.
    """
    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{symbol}: missing columns {missing}")

    if df.empty:
        raise ValueError(f"{symbol}: dataframe is empty")

    if df["date"].isna().any():
        raise ValueError(f"{symbol}: date contains NaT/NaN")

    # OHLC must be numeric and > 0
    for c in ["open", "high", "low", "close"]:
        if df[c].isna().any():
            raise ValueError(f"{symbol}: {c} has NaN")
        if (df[c] <= 0).any():
            bad = df.loc[df[c] <= 0, ["date", "open", "high", "low", "close"]].head(10).to_dict(orient="records")
            raise ValueError(f"{symbol}: {c} <= 0 (examples: {bad})")

    # high >= low
    bad_hl = df["high"] < df["low"]
    if bad_hl.any():
        bad = df.loc[bad_hl, ["date", "open", "high", "low", "close"]].head(10).to_dict(orient="records")
        raise ValueError(f"{symbol}: high < low (examples: {bad})")

    # open/close within [low, high]
    bad_open = (df["open"] < df["low"]) | (df["open"] > df["high"])
    if bad_open.any():
        bad = df.loc[bad_open, ["date", "open", "high", "low", "close"]].head(10).to_dict(orient="records")
        raise ValueError(f"{symbol}: open outside [low, high] (examples: {bad})")

    bad_close = (df["close"] < df["low"]) | (df["close"] > df["high"])
    if bad_close.any():
        bad = df.loc[bad_close, ["date", "open", "high", "low", "close"]].head(10).to_dict(orient="records")
        raise ValueError(f"{symbol}: close outside [low, high] (examples: {bad})")

    # volume must be numeric and >= 0
    vol = pd.to_numeric(df["volume"], errors="coerce")
    if vol.isna().any():
        raise ValueError(f"{symbol}: volume has NaN/non-numeric")
    if (vol < 0).any():
        bad = df.loc[vol < 0, ["date", "volume"]].head(10).to_dict(orient="records")
        raise ValueError(f"{symbol}: volume < 0 (examples: {bad})")


def _build_clean_daily_df(symbol: str, hist: pd.DataFrame) -> pd.DataFrame:
    """
    Convert yfinance history() into Project Purple's clean CSV shape:

      date, symbol, open, high, low, close, volume

    NO junk header rows. NO 'Price' column name.
    """
    needed_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in needed_cols if c not in hist.columns]
    if missing:
        raise ValueError(f"{symbol}: yfinance history missing columns: {missing}")

    df = hist.copy()

    # yfinance index is dates; normalize into a real 'date' column
    df = df.reset_index()

    # Index column can be 'Date' or 'Datetime' depending on interval
    if "Date" in df.columns:
        date_col = "Date"
    elif "Datetime" in df.columns:
        date_col = "Datetime"
    else:
        # fallback: first column is usually the index
        date_col = df.columns[0]

    df["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    df["symbol"] = symbol

    out = pd.DataFrame(
        {
            "date": df["date"],
            "symbol": df["symbol"],
            "open": pd.to_numeric(df["Open"], errors="coerce"),
            "high": pd.to_numeric(df["High"], errors="coerce"),
            "low": pd.to_numeric(df["Low"], errors="coerce"),
            "close": pd.to_numeric(df["Close"], errors="coerce"),
            "volume": pd.to_numeric(df["Volume"], errors="coerce"),
        }
    )

    out = out.dropna(subset=["date", "open", "high", "low", "close", "volume"]).copy()

    # Sort + de-dup (keep last row for a date if duplicates occur)
    out = (
        out.sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )

    _validate_ohlcv(out, symbol)
    return out


def download_and_save_daily_data(
    symbols: Optional[List[str]] = None,
    period: str = "10y",
    interval: str = "1d",
) -> None:
    """
    Download daily OHLCV data using yfinance and save clean CSVs to canonical repo-root /data.

    Behavior:
    - ALWAYS writes clean columns: date,symbol,open,high,low,close,volume
    - If <SYMBOL>_daily.csv exists, we create a timestamped backup first, then replace it.
    - Uses atomic write (tmp -> replace) to avoid half-written files.

    Defaults:
    - If symbols is None, uses strategy_config.candidate_symbols (single source of truth).
    """
    data_dir = _canonical_data_dir()
    use_symbols = [s.upper().strip() for s in (symbols if symbols else list(TICKERS))]

    print(f"Saving CSVs to (canonical): {data_dir}")
    print(f"Symbols to download ({len(use_symbols)}): {use_symbols}")
    print(f"yfinance: period={period}, interval={interval}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for symbol in use_symbols:
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
            clean_df = _build_clean_daily_df(symbol, hist)
        except Exception as e:
            print(f"  ERROR: Could not build validated CSV for {symbol}: {e}")
            continue

        out_path = data_dir / f"{symbol}_daily.csv"
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

        backup_path = None
        if out_path.exists():
            backup_path = data_dir / f"{symbol}_daily.bak_{timestamp}.csv"
            try:
                out_path.replace(backup_path)
            except Exception as e:
                print(f"  ERROR: Could not create backup for {symbol}: {e}")
                continue

        try:
            clean_df.to_csv(tmp_path, index=False)
            tmp_path.replace(out_path)

            last_date = clean_df["date"].iloc[-1].date()
            print(f"  Updated {symbol} safely:")
            if backup_path is not None:
                print(f"    Backup: {backup_path.name}")
            print(f"    Current: {out_path.name}")
            print(f"    Rows: {len(clean_df)} | Last date: {last_date}")

        except PermissionError:
            print(
                f"  ERROR: Permission denied writing {out_path}.\n"
                f"         Close Excel or any program using this file, then run again."
            )
            # try to restore from backup if we made one and the write failed
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass

            if backup_path is not None and backup_path.exists() and not out_path.exists():
                try:
                    backup_path.replace(out_path)
                except Exception:
                    pass

        except Exception as e:
            print(f"  ERROR saving {symbol} to {out_path}: {e}")
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass

            # restore from backup if we made one and the write failed
            if backup_path is not None and backup_path.exists() and not out_path.exists():
                try:
                    backup_path.replace(out_path)
                except Exception:
                    pass


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download daily OHLCV CSVs to repo-root /data (clean format).")
    p.add_argument(
        "--symbols",
        nargs="*",
        default=None,
        help="Optional list of symbols to download (e.g., --symbols SPY AAPL). "
             "If omitted, uses strategy_config.candidate_symbols.",
    )
    p.add_argument("--period", default="10y", help="yfinance period (default: 10y)")
    p.add_argument("--interval", default="1d", help="yfinance interval (default: 1d)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    download_and_save_daily_data(
        symbols=args.symbols,
        period=args.period,
        interval=args.interval,
    )
