from dataclasses import dataclass
from typing import List, Optional, Union

import contextlib
import io

import numpy as np
import pandas as pd

from data_loader import load_symbol_daily
from market_state import get_market_state
from backtest_v2 import (
    run_backtest_for_symbol,
    run_backtest,
    add_atr,
    add_momentum_pullback_signals,
    validate_ohlcv_dataframe,
)
from risk import RiskConfig

# ---------------------------------------------------------------------------
# Universe selection settings (FAST screen)
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

# ---------------------------------------------------------------------------
# Edge-response filter settings (SLOWER, but far more "real")
# ---------------------------------------------------------------------------
# These thresholds are intentionally modest to avoid overfitting and to avoid
# filtering everything out in early development.
EDGE_MIN_TRADES = 30
EDGE_MIN_PROFIT_FACTOR = 1.05
EDGE_MIN_AVG_R = 0.00
EDGE_MAX_DRAWDOWN_PCT = 35.0

EDGE_TEST_INITIAL_EQUITY = 100_000.0
EDGE_TEST_MAX_HOLD_DAYS = 10
EDGE_TEST_TRAIL_ATR_MULTIPLE = 3.0  # keep in sync with main.py default

# Regime behavior
CHOPPY_MAX_UNIVERSE = 5
BULL_MAX_UNIVERSE = 15

BASE_RISK_PER_TRADE_PCT = 0.01  # used only for *recommended* risk in CHOPPY


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
    # fast screen
    symbol: str
    last_date: pd.Timestamp
    last_close: float
    avg_volume: float
    included: bool

    # ranking diagnostics (fast)
    score: float
    ret20: float
    atr_pct: float
    ma_fast: float
    ma_slow: float

    # edge-response metrics (slow)
    edge_trades: int = 0
    edge_win_rate_pct: float = float("nan")
    edge_avg_R: float = float("nan")
    edge_profit_factor: float = float("nan")
    edge_max_dd_pct: float = float("nan")
    edge_score: float = float("nan")
    edge_pass: bool = False
    edge_fail_reason: str = ""

    # regime-aware selection
    risk_mult: float = 1.0
    final_included: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_as_of_date(as_of_date: Optional[Union[str, pd.Timestamp]]) -> Optional[pd.Timestamp]:
    if as_of_date is None:
        return None
    ts = pd.to_datetime(as_of_date, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid as_of_date: {as_of_date!r}")
    return ts


def _slice_as_of(df: pd.DataFrame, as_of_date: Optional[pd.Timestamp]) -> pd.DataFrame:
    """
    Return df filtered to rows with date <= as_of_date (inclusive).
    If as_of_date is None, return df unchanged.
    """
    if as_of_date is None:
        return df
    if "date" not in df.columns:
        raise ValueError("Dataframe missing 'date' column (cannot apply as_of_date cutoff).")
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date"]).copy()
    d = d[d["date"] <= as_of_date].copy()
    d = d.sort_values("date").reset_index(drop=True)
    return d


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
    """
    if not np.isfinite(last_close) or last_close <= 0:
        return float("-inf")

    trend1 = 0.0
    trend2 = 0.0

    if np.isfinite(ma_slow) and ma_slow > 0:
        trend1 = (last_close / ma_slow) - 1.0

    if np.isfinite(ma_fast) and np.isfinite(ma_slow) and ma_slow > 0:
        trend2 = (ma_fast / ma_slow) - 1.0

    vol_penalty = 0.0
    if np.isfinite(atr) and atr > 0:
        vol_penalty = atr / last_close

    # weights chosen to keep the score stable / explainable
    score = (1.0 * trend1) + (0.5 * trend2) - (0.5 * vol_penalty)
    return float(score)


def _profit_factor(pnl: pd.Series) -> float:
    pnl = pnl.dropna()
    if pnl.empty:
        return float("nan")
    wins = pnl[pnl > 0].sum()
    losses = pnl[pnl < 0].sum()  # negative
    if losses == 0:
        return float("inf") if wins > 0 else float("nan")
    return float(wins / abs(losses))


def _max_drawdown_pct(equity_curve: pd.Series) -> float:
    equity_curve = equity_curve.dropna()
    if equity_curve.empty:
        return float("nan")
    running_max = equity_curve.cummax()
    drawdown = (equity_curve / running_max) - 1.0
    return float(drawdown.min() * 100.0)  # negative value


def _edge_score(
    avg_R: float,
    profit_factor: float,
    win_rate_pct: float,
    max_dd_pct: float,
    n_trades: int,
) -> float:
    """
    A simple composite score for ranking. Not a magic number:
    it's just a consistent tie-breaker for "best-to-worst" ordering.

    Higher is better.
    """
    if not np.isfinite(avg_R) or not np.isfinite(win_rate_pct) or not np.isfinite(max_dd_pct):
        return float("nan")

    pf_term = 0.0
    if np.isfinite(profit_factor):
        pf_capped = min(float(profit_factor), 5.0)
        pf_term = (pf_capped - 1.0) * 10.0

    trade_term = min(int(n_trades), 100) * 0.05  # small reward for more samples
    score = (avg_R * 100.0) + pf_term + ((win_rate_pct - 50.0) * 0.2) - (abs(max_dd_pct) * 0.5) + trade_term
    return float(score)


def _run_edge_backtest(
    symbol: str,
    risk_config: RiskConfig,
    as_of_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    HR2 fix:
    - If as_of_date is provided, the edge backtest MUST only use data <= as_of_date.
      This prevents lookahead bias when reusing build_universe() for historical simulation.
    - If as_of_date is None, keep current behavior (full-history edge test).
    """
    # Live / today behavior: keep existing full-history backtest, silence verbose output.
    if as_of_date is None:
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            trades_df, _final_equity = run_backtest_for_symbol(
                symbol=symbol,
                risk_config=risk_config,
                initial_equity=EDGE_TEST_INITIAL_EQUITY,
                max_hold_days=EDGE_TEST_MAX_HOLD_DAYS,
                trail_atr_multiple=EDGE_TEST_TRAIL_ATR_MULTIPLE,
                project_root=None,
            )
        return trades_df

    # Historical (as-of) behavior: build a truncated dataframe and run the core engine directly.
    df = load_symbol_daily(symbol)
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "open", "high", "low", "close"]).reset_index(drop=True)
    df = df.sort_values("date").reset_index(drop=True)

    df = _slice_as_of(df, as_of_date)
    if df.empty:
        return pd.DataFrame()

    # Validate before indicators/backtest
    try:
        validate_ohlcv_dataframe(df=df, symbol=symbol)
    except Exception:
        return pd.DataFrame()

    # Indicators + signal identical to backtest_v2 path
    df = add_atr(df, period=ATR_PERIOD)
    df = add_momentum_pullback_signals(df, ma_fast=MA_FAST, ma_slow=MA_SLOW)
    df = df.dropna(subset=["atr", "ma_fast", "ma_slow"]).reset_index(drop=True)

    if df.empty or len(df) < 5:
        return pd.DataFrame()

    trades_df, _final_equity = run_backtest(
        df=df,
        risk_config=risk_config,
        initial_equity=EDGE_TEST_INITIAL_EQUITY,
        max_hold_days=EDGE_TEST_MAX_HOLD_DAYS,
        trail_atr_multiple=EDGE_TEST_TRAIL_ATR_MULTIPLE,
    )

    if not trades_df.empty:
        trades_df["symbol"] = symbol

    return trades_df


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_symbol(symbol: str, as_of_date: Optional[Union[str, pd.Timestamp]] = None) -> UniverseMember:
    as_of_ts = _normalize_as_of_date(as_of_date)

    df = load_symbol_daily(symbol)
    if df.empty:
        raise ValueError(f"No data for symbol {symbol}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "open", "high", "low", "close", "volume"]).reset_index(drop=True)
    df = df.sort_values("date").reset_index(drop=True)

    # HR2: apply as-of cutoff to prevent using future rows for fast screen too
    df = _slice_as_of(df, as_of_ts)

    if df.empty:
        raise ValueError(f"No usable rows for symbol {symbol} as of {as_of_ts.date() if as_of_ts is not None else 'N/A'}")

    # Indicators (very lightweight)
    df["ma_fast"] = df["close"].rolling(MA_FAST).mean()
    df["ma_slow"] = df["close"].rolling(MA_SLOW).mean()
    df["atr"] = _add_atr(df, period=ATR_PERIOD)
    df["ret20"] = df["close"] / df["close"].shift(RET_LOOKBACK) - 1.0

    last = df.iloc[-1]
    last_close = float(last["close"])
    last_date = pd.to_datetime(last["date"])

    # Liquidity filter over last window (as-of)
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


def build_universe(as_of_date: Optional[Union[str, pd.Timestamp]] = None) -> List[str]:
    """
    Build a tradeable universe in three passes:

    PASS 1 (fast): price + liquidity + simple trend/volatility score.
    PASS 2 (slow): edge-response filter (backtest the *actual* system per symbol).
    PASS 3 (regime-aware): in CHOPPY markets, keep only the best leaders and recommend 1/2 risk.

    HR2 fix:
      - If as_of_date is provided, ALL symbol evaluation and edge-testing is restricted to data <= as_of_date.
        This prevents lookahead bias when using build_universe() for historical simulation.
    """
    as_of_ts = _normalize_as_of_date(as_of_date)

    # --------------------------------------------------------
    # 0) Market regime (SPY)
    # --------------------------------------------------------
    market_regime = "UNKNOWN"
    risk_mult = 1.0

    try:
        # If get_market_state supports as_of_date, use it; otherwise fall back.
        try:
            state = get_market_state(symbol="SPY", as_of_date=as_of_ts)  # type: ignore[arg-type]
        except TypeError:
            state = get_market_state(symbol="SPY")
        market_regime = state.regime
    except Exception as e:
        print("\nWARNING: Could not determine market regime from SPY.")
        print(f"Reason: {e}")
        state = None

    if market_regime == "BULL":
        risk_mult = 1.0
    elif market_regime == "CHOPPY":
        risk_mult = 0.5
    elif market_regime == "BEAR":
        risk_mult = 0.0
    else:
        risk_mult = 1.0

    # --------------------------------------------------------
    # 1) Fast evaluation for all candidates
    # --------------------------------------------------------
    members: List[UniverseMember] = []
    for symbol in CANDIDATE_SYMBOLS:
        try:
            members.append(evaluate_symbol(symbol, as_of_date=as_of_ts))
        except Exception as e:
            print(f"WARNING: could not evaluate {symbol}: {e}")

    if not members:
        print("No symbols evaluated successfully (no data found).")
        return []

    df_all = pd.DataFrame([m.__dict__ for m in members])

    fast_cols = [
        "symbol", "last_date", "last_close", "avg_volume", "included",
        "score", "ret20", "atr_pct", "ma_fast", "ma_slow",
    ]
    df = df_all[fast_cols].copy()

    # Sort by fast score first
    df_sorted_fast = df.sort_values(by="score", ascending=False).reset_index(drop=True)

    df_print_fast = df_sorted_fast.copy()
    df_print_fast["avg_volume"] = df_print_fast["avg_volume"].round(0).astype("Int64")
    df_print_fast["last_close"] = df_print_fast["last_close"].round(2)
    df_print_fast["score"] = df_print_fast["score"].round(4)
    df_print_fast["ret20"] = (df_print_fast["ret20"] * 100.0).round(2)
    df_print_fast["atr_pct"] = (df_print_fast["atr_pct"] * 100.0).round(2)

    print("\n=== PASS 1: Fast screen (price + liquidity + simple score) ===")
    if as_of_ts is not None:
        print(f"(as_of_date cutoff active: <= {as_of_ts.date()})")
    print(df_print_fast[[
        "symbol", "included", "last_close", "avg_volume", "score", "ret20", "atr_pct", "ma_fast", "ma_slow"
    ]].to_string(index=False))

    # If BEAR, we don't want any longs at all.
    if market_regime == "BEAR":
        print(f"\n=== MARKET REGIME: {RED}BEAR{RESET} ===")
        print("Rule: do not trade longs in BEAR. Universe is empty.")
        return []

    # --------------------------------------------------------
    # 2) Edge-response filter (backtest the real system per symbol)
    # --------------------------------------------------------
    risk_config = RiskConfig(
        risk_per_trade_pct=BASE_RISK_PER_TRADE_PCT,
        atr_stop_multiple=1.5,
        atr_target_multiple=3.0,
        max_position_pct_of_equity=0.20,
        min_shares=1,
    )

    included_symbols = df_sorted_fast[df_sorted_fast["included"] == True]["symbol"].tolist()

    edge_rows = []
    for sym in included_symbols:
        trades_df = _run_edge_backtest(sym, risk_config=risk_config, as_of_date=as_of_ts)

        n_trades = int(len(trades_df))
        if n_trades == 0:
            win_rate = float("nan")
            avg_R = float("nan")
            pf = float("nan")
            max_dd = float("nan")
            edge_pass = False
            reason = "no_trades"
            score = float("nan")
        else:
            win_rate = float((trades_df["pnl_dollars"] > 0).mean() * 100.0)
            avg_R = float(trades_df["R"].mean())
            pf = _profit_factor(trades_df["pnl_dollars"])
            equity_curve = pd.concat(
                [pd.Series([EDGE_TEST_INITIAL_EQUITY]),
                 trades_df["equity_after"].reset_index(drop=True)],
                ignore_index=True,
            )
            max_dd = _max_drawdown_pct(equity_curve)  # negative
            score = _edge_score(avg_R=avg_R, profit_factor=pf, win_rate_pct=win_rate, max_dd_pct=max_dd, n_trades=n_trades)

            # Pass/fail rules
            reasons = []
            if n_trades < EDGE_MIN_TRADES:
                reasons.append(f"trades<{EDGE_MIN_TRADES}")
            if np.isfinite(avg_R) and avg_R < EDGE_MIN_AVG_R:
                reasons.append("avg_R<0")
            if np.isfinite(pf) and pf < EDGE_MIN_PROFIT_FACTOR:
                reasons.append(f"pf<{EDGE_MIN_PROFIT_FACTOR}")
            if np.isfinite(max_dd) and abs(max_dd) > EDGE_MAX_DRAWDOWN_PCT:
                reasons.append(f"max_dd>{EDGE_MAX_DRAWDOWN_PCT}%")

            edge_pass = (len(reasons) == 0)
            reason = "OK" if edge_pass else ";".join(reasons)

        edge_rows.append({
            "symbol": sym,
            "edge_trades": n_trades,
            "edge_win_rate_pct": win_rate,
            "edge_avg_R": avg_R,
            "edge_profit_factor": pf,
            "edge_max_dd_pct": max_dd,
            "edge_score": score,
            "edge_pass": edge_pass,
            "edge_fail_reason": reason,
        })

    df_edge = pd.DataFrame(edge_rows)
    df2 = df_sorted_fast.merge(df_edge, on="symbol", how="left")

    # fill non-tested symbols (excluded by fast screen)
    df2["edge_trades"] = df2["edge_trades"].fillna(0).astype(int)
    for col in ["edge_win_rate_pct", "edge_avg_R", "edge_profit_factor", "edge_max_dd_pct", "edge_score"]:
        df2[col] = pd.to_numeric(df2[col], errors="coerce")

    # FIX: avoid pandas FutureWarning by converting dtype BEFORE fillna
    df2["edge_pass"] = df2["edge_pass"].astype("boolean").fillna(False).astype(bool)
    df2["edge_fail_reason"] = df2["edge_fail_reason"].fillna("not_tested")

    df_print_edge = df2.copy()
    df_print_edge["edge_win_rate_pct"] = df_print_edge["edge_win_rate_pct"].round(1)
    df_print_edge["edge_avg_R"] = df_print_edge["edge_avg_R"].round(3)
    df_print_edge["edge_profit_factor"] = df_print_edge["edge_profit_factor"].replace([np.inf], 999.0).round(2)
    df_print_edge["edge_max_dd_pct"] = df_print_edge["edge_max_dd_pct"].round(1)
    df_print_edge["edge_score"] = df_print_edge["edge_score"].round(2)

    print("\n=== PASS 2: Edge-response filter (per-symbol backtest metrics) ===")
    print(df_print_edge[[
        "symbol", "included", "edge_pass", "edge_fail_reason",
        "edge_score", "edge_trades", "edge_avg_R", "edge_profit_factor", "edge_win_rate_pct", "edge_max_dd_pct"
    ]].sort_values(by=["edge_pass", "edge_score"], ascending=[False, False]).to_string(index=False))

    # --------------------------------------------------------
    # 3) Regime-aware selection (CHOPPY stricter, 1/2 risk recommended)
    # --------------------------------------------------------
    if market_regime == "CHOPPY":
        print(f"\n=== MARKET REGIME: {YELLOW}CHOPPY{RESET} ===")
        print("Rules:")
        print("  - Keep only edge-pass symbols")
        print("  - Stricter selection: ret20 > 0 AND ma_fast > ma_slow")
        print(f"  - Keep top {CHOPPY_MAX_UNIVERSE} by edge_score")
        print(f"  - Recommended risk per trade: {BASE_RISK_PER_TRADE_PCT * risk_mult:.3%} (1/2 of normal)")

        choppy_mask = (
            (df2["included"] == True) &
            (df2["edge_pass"] == True) &
            (df2["ret20"] > 0) &
            (df2["ma_fast"] > df2["ma_slow"])
        )
        df_final = df2[choppy_mask].copy()
        df_final = df_final.sort_values(by="edge_score", ascending=False).head(CHOPPY_MAX_UNIVERSE)

    else:
        regime_str = f"{GREEN}BULL{RESET}" if market_regime == "BULL" else f"{YELLOW}{market_regime}{RESET}"
        print(f"\n=== MARKET REGIME: {regime_str} ===")
        print("Rules:")
        print("  - Keep only edge-pass symbols")
        print(f"  - Keep top {BULL_MAX_UNIVERSE} by edge_score")
        print(f"  - Recommended risk per trade: {BASE_RISK_PER_TRADE_PCT * risk_mult:.3%}")

        bull_mask = (df2["included"] == True) & (df2["edge_pass"] == True)
        df_final = df2[bull_mask].copy()
        df_final = df_final.sort_values(by="edge_score", ascending=False).head(BULL_MAX_UNIVERSE)

    universe = df_final["symbol"].tolist()

    # Pretty final table (ranked)
    if df_final.empty:
        print("\n=== PASS 3: Final universe (ranked) ===")
        print("No symbols passed the edge filter + regime rules.")
        return []

    df_final_print = df_final.copy()
    df_final_print["rank"] = np.arange(1, len(df_final_print) + 1)
    df_final_print["last_close"] = df_final_print["last_close"].round(2)
    df_final_print["ret20"] = (df_final_print["ret20"] * 100.0).round(2)
    df_final_print["atr_pct"] = (df_final_print["atr_pct"] * 100.0).round(2)
    df_final_print["score"] = df_final_print["score"].round(4)
    df_final_print["edge_score"] = df_final_print["edge_score"].round(2)
    df_final_print["edge_avg_R"] = df_final_print["edge_avg_R"].round(3)
    df_final_print["edge_profit_factor"] = df_final_print["edge_profit_factor"].replace([np.inf], 999.0).round(2)
    df_final_print["edge_win_rate_pct"] = df_final_print["edge_win_rate_pct"].round(1)
    df_final_print["edge_max_dd_pct"] = df_final_print["edge_max_dd_pct"].round(1)

    print("\n=== PASS 3: Final universe (ranked) ===")
    print(df_final_print[[
        "rank", "symbol",
        "edge_score", "edge_trades", "edge_avg_R", "edge_profit_factor", "edge_win_rate_pct", "edge_max_dd_pct",
        "score", "ret20", "atr_pct"
    ]].to_string(index=False))

    print("\n=== Top 5 picks ===")
    print(df_final_print.head(5)[[
        "rank", "symbol", "edge_score", "edge_avg_R", "edge_profit_factor", "edge_trades"
    ]].to_string(index=False))

    print("\n=== Final trading universe ===")
    print(universe)

    return universe


if __name__ == "__main__":
    build_universe()
