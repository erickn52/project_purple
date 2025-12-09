"""
scanner_simple.py

Simple momentum scanner with:
  - 20-day and 60-day momentum
  - Composite momentum score (higher = stronger)
  - Color-coded scores (green / yellow / red)
  - Trend classification:
        +/+  = Strong uptrend
        +/–  = New reversal / early breakout
        –/+  = Pullback in an uptrend
        –/–  = Strong downtrend
"""

from typing import List, Dict
import pandas as pd
from data_loader import load_symbol_daily, TICKERS

LOOKBACK_SHORT = 20
LOOKBACK_LONG = 60
TOP_N_DEFAULT = 10

# ANSI color codes for terminal
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"


def color_score(score: float) -> str:
    """
    Color the composite score:
      green  = strong (>= 0.60)
      yellow = neutral (0.40–0.59)
      red    = weak   (< 0.40)
    """
    if score >= 0.60:
        return f"{GREEN}{score:.2f}{RESET}"
    elif score >= 0.40:
        return f"{YELLOW}{score:.2f}{RESET}"
    else:
        return f"{RED}{score:.2f}{RESET}"


def classify_trend(ret20: float, ret60: float) -> str:
    """
    Classify the momentum pattern based on signs of ret20 and ret60.

        ret20 > 0, ret60 > 0  -> Strong uptrend
        ret20 > 0, ret60 <= 0 -> New reversal / early breakout
        ret20 <= 0, ret60 > 0 -> Pullback in an uptrend
        ret20 <= 0, ret60 <= 0-> Strong downtrend
    """
    if ret20 > 0 and ret60 > 0:
        return "Strong uptrend"
    elif ret20 > 0 and ret60 <= 0:
        return "Reversal / early breakout"
    elif ret20 <= 0 and ret60 > 0:
        return "Pullback in uptrend"
    else:
        return "Strong downtrend"


def build_momentum_table() -> pd.DataFrame:
    """
    Compute ret20 and ret60 for each ticker in TICKERS.
    """
    rows: List[Dict] = []

    for symbol in TICKERS:
        df = load_symbol_daily(symbol)

        if len(df) <= LOOKBACK_LONG:
            print(f"Skipping {symbol}: not enough data ({len(df)} rows).")
            continue

        close = df["close"]
        last = float(close.iloc[-1])
        past20 = float(close.iloc[-1 - LOOKBACK_SHORT])
        past60 = float(close.iloc[-1 - LOOKBACK_LONG])

        ret20 = last / past20 - 1.0
        ret60 = last / past60 - 1.0

        rows.append({"symbol": symbol, "ret20": ret20, "ret60": ret60})

    return pd.DataFrame(rows)


def rank_momentum(df: pd.DataFrame, top_n: int = TOP_N_DEFAULT) -> pd.DataFrame:
    """
    Add percentile ranks and composite score, then classify trend type.
    Higher score = stronger momentum.
    """

    # Percentile ranks: ascending=True -> lowest return ~0.1, highest return = 1.0
    df["rank_ret20"] = df["ret20"].rank(ascending=True, pct=True)
    df["rank_ret60"] = df["ret60"].rank(ascending=True, pct=True)

    # Composite score emphasizing recent momentum
    df["score"] = 0.6 * df["rank_ret20"] + 0.4 * df["rank_ret60"]

    # Trend classification
    df["trend"] = df.apply(
        lambda row: classify_trend(row["ret20"], row["ret60"]), axis=1
    )

    # Sort BEST → WORST by score
    df = df.sort_values("score", ascending=False)

    return df.head(top_n)


def run_simple_scanner(top_n: int = TOP_N_DEFAULT):
    print("Running SIMPLE momentum scanner on universe:")
    print(TICKERS)

    df = build_momentum_table()

    # Display raw momentum sorted best → worst
    print("\n=== Raw Momentum (20-day BEST → WORST) ===")
    print(
        df.sort_values("ret20", ascending=False)[["symbol", "ret20"]]
        .round(4)
        .to_string(index=False)
    )

    print("\n=== Raw Momentum (60-day BEST → WORST) ===")
    print(
        df.sort_values("ret60", ascending=False)[["symbol", "ret60"]]
        .round(4)
        .to_string(index=False)
    )

    ranked = rank_momentum(df, top_n)

    print(f"\n=== Top {top_n} Composite Momentum Scores (BEST → WORST) ===")
    print("symbol  score   ret20     ret60      trend")
    for _, row in ranked.iterrows():
        sym = row["symbol"]
        score_colored = color_score(row["score"])
        r20 = row["ret20"]
        r60 = row["ret60"]
        trend = row["trend"]
        print(
            f"{sym:5}  {score_colored:7}  {r20:7.4f}  {r60:8.4f}  {trend}"
        )


if __name__ == "__main__":
    run_simple_scanner(TOP_N_DEFAULT)
