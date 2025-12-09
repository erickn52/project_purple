import yfinance as yf
import pandas as pd
from pathlib import Path


# List of tickers we want daily data for
# (expanded to include AMZN, IBKR, AMDL per your request)
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


def download_and_save_daily_data():
    """
    Download daily OHLCV data for each ticker using yfinance and save to CSV
    in the same format as your existing AAPL_daily.csv:

        columns: Price, symbol, open, high, low, close, volume

    Data rows:
        Row 0: Ticker,<sym>,<sym>,...,<sym>
        Row 1: date,,,,,,
        Row 2+: YYYY-MM-DD, <sym>, open, high, low, close, volume
    """

    # project_root = .../project_purple
    # __file__ = .../project_purple/project_purple/data_downloader.py
    # parents[0] -> .../project_purple/project_purple
    # parents[1] -> .../project_purple
    project_root = Path(__file__).resolve().parents[1]

    # .../project_purple/project_purple/data
    data_dir = project_root / "project_purple" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving CSVs to: {data_dir}")

    for symbol in TICKERS:
        print(f"\nDownloading {symbol} daily data with yfinance...")

        # 10 years of daily data; adjust as you like
        hist = yf.download(
            symbol,
            period="10y",
            interval="1d",
            auto_adjust=False,
            progress=False,
        )

        if hist.empty:
            print(f"  WARNING: No data returned for {symbol}. "
                  f"Check if the ticker is correct or too illiquid.")
            continue

        # If yfinance gives MultiIndex columns, flatten them into single strings
        if isinstance(hist.columns, pd.MultiIndex):
            flat_cols = []
            for col in hist.columns:
                # col might be like ('AAPL', 'Open') or ('Close', '')
                parts = [str(x) for x in col if x != ""]
                flat_cols.append("_".join(parts))
            hist.columns = flat_cols

        # Bring the date index out as a column
        hist = hist.reset_index()

        # Make sure the first column is the date
        first_col = hist.columns[0]
        if first_col != "Date":
            hist = hist.rename(columns={first_col: "Date"})

        # All other columns are price-related; we'll use them by position, not name
        price_cols = [c for c in hist.columns if c != "Date"]

        if len(price_cols) < 5:
            print(
                f"  WARNING: For {symbol}, expected at least 5 price columns, "
                f"got {len(price_cols)}. Columns are: {list(hist.columns)}"
            )
            continue

        # We assume the order from yfinance is roughly:
        # [Open, High, Low, Close, (Adj Close), Volume]
        # We'll map them by position:
        open_col = price_cols[0]
        high_col = price_cols[1]
        low_col = price_cols[2]
        close_col = price_cols[3]
        volume_col = price_cols[-1]  # last one is usually Volume

        # Convert to clean 1-D Series
        date_series = pd.to_datetime(hist["Date"]).dt.strftime("%Y-%m-%d")
        open_series = pd.Series(hist[open_col].to_numpy().ravel())
        high_series = pd.Series(hist[high_col].to_numpy().ravel())
        low_series = pd.Series(hist[low_col].to_numpy().ravel())
        close_series = pd.Series(hist[close_col].to_numpy().ravel())
        volume_series = pd.Series(hist[volume_col].to_numpy().ravel())

        # Body: real daily data rows
        body = pd.DataFrame(
            {
                "Price": date_series,
                "symbol": [symbol] * len(hist),
                "open": open_series,
                "high": high_series,
                "low": low_series,
                "close": close_series,
                "volume": volume_series,
            }
        )

        # Header rows as DATA
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

        # Combine header rows and body
        final_df = pd.concat([header_rows, body], ignore_index=True)

        # Ensure correct column order / names
        final_df = final_df[
            ["Price", "symbol", "open", "high", "low", "close", "volume"]
        ]

        out_path = data_dir / f"{symbol}_daily.csv"

        # Try to save, but handle file-lock issues gracefully
        try:
            final_df.to_csv(out_path, index=False)
            print(f"  Saved {symbol} to: {out_path}")
        except PermissionError:
            print(
                f"  ERROR: Permission denied writing {out_path}.\n"
                f"         Close Excel or any program using this file, "
                f"then run the script again."
            )
        except Exception as e:
            print(f"  ERROR saving {symbol} to {out_path}: {e}")


if __name__ == "__main__":
    download_and_save_daily_data()
