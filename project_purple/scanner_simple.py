from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from data_loader import load_symbol_daily

# ---------------------------------------------------------------------------
# Universe selection settings
# ---------------------------------------------------------------------------

MIN_PRICE = 15.0
MAX_PRICE = 125.0

VOLUME_LOOKBACK = 20
MIN_AVG_VOLUME = 1_000_000

# Signal / score settings (lightweight + fast)
MA_FAST = 20
MA_SLOW = 50
ATR_PERIOD = 14
RET_LOOKBACK = 20  # optional: helps ranking

CANDIDATE_SYMBOLS: List[str] = [
    "AAPL", "AMD", "AMDL", "AMZN", "IBKR",
    "META", "MSFT", "NVDA", "SPY", "TSLA",

    "PLTR", "SHOP", "UBER", "SNAP", "PINS",
    "ZM", "CRWD", "NET", "PATH",
    "OKTA", "HUBS", "DDOG",

    "DKNG", "CROX", "LULU", "ETSY", "RBLX",
    "CELH", "PTON", "ROKU", "FVRR", "LYFT",

    "RIVN", "LCID", "RUN", "ENPH", "FSLR",

    "SOFI", "COIN", "HOOD",

    "NVAX", "BNTX", "MRNA", "IONS", "REGN",

    "BA", "GE", "UAL", "DAL",

    "WBD", "DIS",

    "OXY", "APA", "FCX", "AA",

    "AFRM", "MDB", "ZS", "TEAM",
]

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
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

    # ranking diagnostics
    score: float
    ret20: float
    atr_pct: float
    ma_fast: float
    ma_slow: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _compute_score(last_close: float, ma_fast: float, ma_slow: float, atr: float) -> float:
    """
    Simple, explainable ranking score.

    Higher is better. Intuition:
      + prefer trend strength (close above slow MA; fast MA above slow MA)
      - penalize very volatile names (ATR as % of price)

    This is NOT ML. It’s a deterministic tie-breaker.
    """
    if not np.isfinite(last_close) or last_close <= 0:
        return -np.inf
    if not np.isfinite(ma_slow) or ma_slow <= 0:
        return -np.inf
    if not np.isfinite(ma_fast) or ma_fast <= 0:
        return -np.inf
    if not np.isfinite(atr) or atr <= 0:
        return -np.inf

    trend1 = (last_close / ma_slow) - 1.0
    trend2 = (ma_fast / ma_slow) - 1.0
    vol_penalty = (atr / last_close)

    # weights chosen to keep it stable / not overly sensitive
    score = (1.0 * trend1) + (0.5 * trend2) - (0.5 * vol_penalty)
    return float(score)


# ---------------------------------------------------------------------------
# Core universe evaluation
# ---------------------------------------------------------------------------

def evaluate_symbol(symbol: str) -> UniverseMember:
    df = load_symbol_daily(symbol)
    if df.empty:
        raise ValueError(f"No data for symbol {symbol}")

    df = df.copy()

    # Make sure date is datetime for safety (load_symbol_daily should already do this)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "open", "high", "low", "close", "volume"]).reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No usable rows for symbol {symbol}")

    # Indicators (very lightweight)
    df["ma_fast"] = df["close"].rolling(MA_FAST).mean()
    df["ma_slow"] = df["close"].rolling(MA_SLOW).mean()
    df["atr"] = _add_atr(df, period=ATR_PERIOD)
    df["ret20"] = df["close"] / df["close"].shift(RET_LOOKBACK) - 1.0

    last = df.iloc[-1]
    last_close = float(last["close"])
    last_date = pd.to_datetime(last["date"])

    # Liquidity filter over last window
    vol_window = df["volume"].tail(VOLUME_LOOKBACK)
    avg_volume = float(vol_window.mean()) if len(vol_window) > 0 else np.nan

    price_ok = (MIN_PRICE <= last_close <= MAX_PRICE)
    volume_ok = (not np.isnan(avg_volume)) and (avg_volume >= MIN_AVG_VOLUME)
    included = bool(price_ok and volume_ok)

    ma_fast = float(last["ma_fast"]) if np.isfinite(last["ma_fast"]) else np.nan
    ma_slow = float(last["ma_slow"]) if np.isfinite(last["ma_slow"]) else np.nan
    atr = float(last["atr"]) if np.isfinite(last["atr"]) else np.nan
    ret20 = float(last["ret20"]) if np.isfinite(last["ret20"]) else np.nan
    atr_pct = float(atr / last_close) if np.isfinite(atr) and last_close > 0 else np.nan

    score = _compute_score(last_close=last_close, ma_fast=ma_fast, ma_slow=ma_slow, atr=atr)

    return UniverseMember(
        symbol=symbol,
        last_date=last_date,
        last_close=last_close,
        avg_volume=avg_volume,
        included=included,
        score=score,
        ret20=ret20,
        atr_pct=atr_pct,
        ma_fast=ma_fast,
        ma_slow=ma_slow,
    )


def build_universe() -> List[str]:
    members: List[UniverseMember] = []

    for symbol in CANDIDATE_SYMBOLS:
        try:
            members.append(evaluate_symbol(symbol))
        except Exception as e:
            print(f"WARNING: could not evaluate {symbol}: {e}")

    if not members:
        print("No symbols evaluated successfully (no data found).")
        return []

    df = pd.DataFrame([m.__dict__ for m in members])

    # Sort:
    #   1) included first
    #   2) score descending (best first)
    #   3) last_close ascending (tie-break)
    df_sorted = df.sort_values(
        by=["included", "score", "last_close"],
        ascending=[False, False, True],
    )

    df_print = df_sorted.copy()
    df_print["included"] = df_print["included"].map(lambda x: f"{GREEN}True{RESET}" if x else f"{RED}False{RESET}")

    df_print["last_close"] = df_print["last_close"].round(2)
    df_print["avg_volume"] = df_print["avg_volume"].round(0)
    df_print["score"] = df_print["score"].round(5)
    df_print["ret20"] = df_print["ret20"].round(4)
    df_print["atr_pct"] = (df_print["atr_pct"] * 100.0).round(2)

    print("\n=== Universe evaluation (sorted) ===")
    print(df_print.to_string(index=False))

    universe = [row["symbol"] for _, row in df_sorted.iterrows() if bool(row["included"])]

    print("\n=== Final trading universe ===")
    print(universe if universe else "No symbols passed the filters.")

    return universe


if __name__ == "__main__":
    build_universe()
