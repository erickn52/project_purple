from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from data_loader import load_symbol_daily

# ---------------------------------------------------------------------------
# Universe selection settings
# ---------------------------------------------------------------------------

# Price range we actually want to trade
MIN_PRICE = 15.0
MAX_PRICE = 125.0

# Minimum average daily volume over the last N bars
VOLUME_LOOKBACK = 20
MIN_AVG_VOLUME = 750_000  # adjust later once you see how many pass

# Curated list of candidate symbols.
# This includes your original tickers plus a broader set of
# swing-trade-friendly names across sectors. The price filter
# below (15–125) will decide which ones are currently tradable.
CANDIDATE_SYMBOLS: List[str] = [
    # --- Your original symbols ---
    "AAPL", "AMD", "AMDL", "AMZN", "IBKR",
    "META", "MSFT", "NVDA", "SPY", "TSLA",

    # --- Technology / Growth ---
    "PLTR", "SHOP", "UBER", "SNAP", "PINS",
    "ZM", "CRWD", "NET", "PATH",
    "OKTA", "HUBS", "DDOG",

    # --- Consumer / Retail / Discretionary ---
    "DKNG", "CROX", "LULU", "ETSY", "RBLX",
    "CELH", "PTON", "ROKU", "FVRR", "LYFT",

    # --- EV / Clean Energy / Solar ---
    "RIVN", "LCID", "RUN", "ENPH", "FSLR",

    # --- Finance / Crypto-adjacent ---
    "SOFI", "COIN", "HOOD",

    # --- Healthcare / Biotech ---
    "NVAX", "BNTX", "MRNA", "IONS", "REGN",

    # --- Industrials / Airlines ---
    "BA", "GE", "UAL", "DAL",

    # --- Communication / Media ---
    "WBD", "DIS",

    # --- Energy / Materials ---
    "OXY", "APA", "FCX", "AA",

    # --- High-momentum / software names ---
    "AFRM", "MDB", "ZS", "TEAM",
]


# ANSI color codes for console output (PyCharm run window supports these)
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class UniverseMember:
    symbol: str
    last_date: pd.Timestamp
    last_close: float
    avg_volume: float
    included: bool


# ---------------------------------------------------------------------------
# Core universe evaluation
# ---------------------------------------------------------------------------


def evaluate_symbol(symbol: str) -> UniverseMember:
    """
    Load historical data for a symbol and decide if it belongs in today's
    trading universe based on price and liquidity filters.

    Filters:
      - last closing price between MIN_PRICE and MAX_PRICE
      - average volume over last VOLUME_LOOKBACK bars >= MIN_AVG_VOLUME
    """
    df = load_symbol_daily(symbol)

    if df.empty:
        raise ValueError(f"No data for symbol {symbol}")

    df = df.copy()

    # Use the last row as the "current" bar
    last = df.iloc[-1]
    last_close = float(last["close"])
    last_date = pd.to_datetime(last["date"])

    # Compute volume filter over the recent window
    vol_window = df["volume"].tail(VOLUME_LOOKBACK)
    avg_volume = float(vol_window.mean()) if len(vol_window) > 0 else np.nan

    # Apply simple filters
    price_ok = (MIN_PRICE <= last_close <= MAX_PRICE)
    volume_ok = (not np.isnan(avg_volume)) and (avg_volume >= MIN_AVG_VOLUME)

    included = bool(price_ok and volume_ok)

    return UniverseMember(
        symbol=symbol,
        last_date=last_date,
        last_close=last_close,
        avg_volume=avg_volume,
        included=included,
    )


def build_universe() -> List[str]:
    """
    Evaluate all symbols in CANDIDATE_SYMBOLS and return a list of those that
    pass the filters (price + volume).

    This will be the list your backtests and live system use going forward.
    """
    members: List[UniverseMember] = []

    for symbol in CANDIDATE_SYMBOLS:
        try:
            member = evaluate_symbol(symbol)
            members.append(member)
        except Exception as e:
            # This will happen for symbols we have not downloaded yet.
            print(f"WARNING: could not evaluate {symbol}: {e}")

    if members:
        df = pd.DataFrame([m.__dict__ for m in members])

        # Sort so that:
        # - included=True first (i.e., included descending)
        # - within each group, sort by last_close ascending (cheapest first)
        df_sorted = df.sort_values(
            by=["included", "last_close"],
            ascending=[False, True],
        )

        # Build a copy for pretty printing, with colored True/False
        df_print = df_sorted.copy()
        df_print["included"] = df_print["included"].map(
            lambda x: f"{GREEN}True{RESET}" if x else f"{RED}False{RESET}"
        )

        # Optionally round prices/volume for nicer display
        df_print["last_close"] = df_print["last_close"].round(2)
        df_print["avg_volume"] = df_print["avg_volume"].round(0)

        print("\n=== Universe evaluation (sorted) ===")
        print(df_print.to_string(index=False))
    else:
        print("No symbols evaluated successfully (no data found).")

    # Return only the included symbols (True/False logic unchanged)
    universe = [m.symbol for m in members if m.included]

    print("\n=== Final trading universe ===")
    if universe:
        print(universe)
    else:
        print("No symbols passed the filters.")

    return universe


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    build_universe()
