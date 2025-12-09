"""
scanner_simple.py

B1 – Simple momentum-only scanner.

This script:
  - Loads daily data for all symbols in TICKERS
  - Computes:
      * 20-day return (ret20)
      * 60-day return (ret60)
  - Ranks symbols by a composite momentum score
  - Prints a table of results and the top N symbols
"""

from typing import Dict, List

import pandas as pd

from data_loader import load_symbol_daily, TICKERS


# Lookback windows for momentum
LOOKBACK_SHORT = 20   # about 1 trading month
LOOKBACK_LONG = 60    # about 3 trading months

TOP_N_DEFAULT = 10    # how many names to show as "best"


def build_momentum_table() -> pd.DataFrame:
    """
    For each symbol in TICKERS, compute simple momentum stats:

        ret20 = last_close / close_20_days_ago - 1
        ret60 = last_close / close_60_days_ago - 1

    Returns a DataFrame with one row per symbol.
    """
    rows: List[Dict] = []

    for symbol in TICKERS:
        df = load_symbol_daily(symbol)

        # Need at least 60 bars to compute 60-day momentum safely
        if len(df) <= LOOKBACK_LONG:
            print(f"Skipping {symbol}: not enough history ({len(df)} rows).")
            continue

        close = df["close"]

        last_close = float(close.iloc[-1])
        past_close_20 = float(close.iloc[-1 - LOOKBACK_SHORT])
        past_close_60 = float(close.iloc[-1 - LOOKBACK_LONG])

        ret20 = last_close / past_close_20 - 1.0
        ret60 = last_close / past_close_60 - 1.0

        rows.append(
            {
                "symbol": symbol,
                "ret20": ret20,
                "ret60": ret60,
            }
        )

    if not rows:
        raise ValueError("No symbols had enough data for momentum computation.")

    mom_df = pd.DataFrame(rows)

    return mom_df


def rank_momentum(mom_df: pd.DataFrame, top_n: int = TOP_N_DEFAULT) -> pd.DataFrame:
    """
    Rank symbols by a simple composite momentum score.

    We combine 20-day and 60-day returns:

        score = 0.6 * percentile(ret20) + 0.4 * percentile(ret60)

    where percentile() is the rank in [0, 1].

    - Emphasizes recent momentum (ret20),
    - Still respects the medium-term trend (ret60).
    """
    df = mom_df.copy()

    # Percentile ranks: higher return => higher percentile
    df["rank_ret20"] = df["ret20"].rank(ascending=False, pct=True)
    df["rank_ret60"] = df["ret60"].rank(ascending=False, pct=True)

    # Composite score
    df["score"] = 0.6 * df["rank_ret20"] + 0.4 * df["rank_ret60"]

    # Sort by score descending (strongest momentum first)
    df = df.sort_values("score", ascending=False)

    return df.head(top_n)


def run_simple_scanner(top_n: int = TOP_N_DEFAULT):
    print("Running SIMPLE momentum scanner on universe:")
    print(TICKERS)

    mom_df = build_momentum_table()

    print("\n=== Raw Momentum (returns as decimals) ===")
    print(
        mom_df[["symbol", "ret20", "ret60"]]
        .round(4)
        .to_string(index=False)
    )

    ranked = rank_momentum(mom_df, top_n=top_n)

    print(f"\n=== Top {top_n} by simple momentum score ===")
    print(
        ranked[["symbol", "score", "ret20", "ret60"]]
        .round(4)
        .to_string(index=False)
    )


if __name__ == "__main__":
    run_simple_scanner(TOP_N_DEFAULT)
