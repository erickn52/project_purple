from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


# --- Make repo-root importable when running: python .\tests\test_data_integrity.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from project_purple.data_loader import get_data_dir  # noqa: E402


@dataclass
class FileCheckResult:
    path: Path
    rows: int
    bad_high_low: int
    bad_open_range: int
    bad_close_range: int
    bad_volume: int
    duplicate_dates: int


def _infer_symbol_from_filename(path: Path) -> str:
    """
    Handles:
      - AAPL_daily.csv
      - SPY_daily.download_20251224_162926.csv
    """
    name = path.name.upper()
    if "_DAILY" not in name:
        return path.stem.split("_")[0]
    return name.split("_DAILY")[0]


def _load_and_clean_csv(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)

    # Standardize date column name
    if "Price" in raw.columns and "date" not in raw.columns:
        raw = raw.rename(columns={"Price": "date"})

    required = {"date", "symbol", "open", "high", "low", "close", "volume"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = raw.copy()

    # Remove junk header rows that appear inside the CSV
    date_str = df["date"].astype(str).str.strip().str.lower()
    bad_tokens = {"ticker", "date", "", "nan", "none"}
    df = df[~date_str.isin(bad_tokens)].copy()

    # Parse dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()

    # Numeric conversion
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with missing price data
    df = df.dropna(subset=["open", "high", "low", "close"]).copy()

    # Normalize types/order
    df = df[["date", "symbol", "open", "high", "low", "close", "volume"]].copy()

    # Sort + de-dup dates
    df = df.sort_values("date").reset_index(drop=True)

    return df


def _validate_ohlcv(df: pd.DataFrame) -> Tuple[int, int, int, int, int]:
    """
    Returns:
      (bad_high_low, bad_open_range, bad_close_range, bad_volume, duplicate_dates)
    """
    # High must be >= Low
    bad_high_low = int((df["high"] < df["low"]).sum())

    # Open/Close should fall within [low, high] (allow exact endpoints)
    bad_open_range = int(((df["open"] < df["low"]) | (df["open"] > df["high"])).sum())
    bad_close_range = int(((df["close"] < df["low"]) | (df["close"] > df["high"])).sum())

    # Volume should be >= 0 (or NaN is already dropped by numeric conversion + price checks)
    bad_volume = int((df["volume"] < 0).sum())

    # Duplicate dates
    duplicate_dates = int(df["date"].duplicated().sum())

    return bad_high_low, bad_open_range, bad_close_range, bad_volume, duplicate_dates


def main() -> None:
    data_dir = get_data_dir()
    print(f"[DATA INTEGRITY] Using data dir: {data_dir}")

    if not data_dir.exists():
        raise RuntimeError(f"Canonical data dir does not exist: {data_dir}")

    # Validate both canonical files and any downloader "download_" files
    csvs = sorted(list(data_dir.glob("*_daily.csv")) + list(data_dir.glob("*_daily.download_*.csv")))
    if not csvs:
        raise RuntimeError(f"No daily CSVs found in: {data_dir}")

    results: List[FileCheckResult] = []
    failures: List[str] = []

    for path in csvs:
        symbol = _infer_symbol_from_filename(path)

        try:
            df = _load_and_clean_csv(path)

            # Optional: make sure the symbol column matches what the file claims
            # (We don't hard-fail on mixed casing, but do on totally wrong symbols.)
            sym_col = df["symbol"].astype(str).str.upper().unique().tolist()
            if sym_col and symbol not in sym_col:
                failures.append(f"{path.name}: symbol mismatch (filename {symbol}, column has {sym_col[:5]})")

            bad_high_low, bad_open_range, bad_close_range, bad_volume, duplicate_dates = _validate_ohlcv(df)

            results.append(
                FileCheckResult(
                    path=path,
                    rows=len(df),
                    bad_high_low=bad_high_low,
                    bad_open_range=bad_open_range,
                    bad_close_range=bad_close_range,
                    bad_volume=bad_volume,
                    duplicate_dates=duplicate_dates,
                )
            )

            # Hard failures
            if bad_high_low > 0:
                failures.append(f"{path.name}: high<low rows={bad_high_low}")

            # These are “strong warnings” (usually indicates corrupted OHLCV)
            if bad_open_range > 0:
                failures.append(f"{path.name}: open outside [low,high] rows={bad_open_range}")
            if bad_close_range > 0:
                failures.append(f"{path.name}: close outside [low,high] rows={bad_close_range}")

            if bad_volume > 0:
                failures.append(f"{path.name}: negative volume rows={bad_volume}")

            if duplicate_dates > 0:
                failures.append(f"{path.name}: duplicate dates rows={duplicate_dates}")

        except Exception as e:
            failures.append(f"{path.name}: exception: {e}")

    total_files = len(results)
    total_rows = sum(r.rows for r in results)
    total_bad_hl = sum(r.bad_high_low for r in results)

    print(f"[DATA INTEGRITY] Checked files: {total_files} | total rows: {total_rows} | high<low total: {total_bad_hl}")

    if failures:
        print("\n=== DATA INTEGRITY FAILURES ===")
        for msg in failures:
            print(f"- {msg}")
        raise SystemExit(1)

    print("\nOK: All checked CSVs passed OHLCV integrity checks.")


if __name__ == "__main__":
    main()
