from __future__ import annotations

from typing import Tuple

import pandas as pd

from project_purple.settings import strategy_settings


def run_simple_backtest(
    df: pd.DataFrame,
    symbol: str,
    initial_equity: float = 100_000.0,
) -> Tuple[pd.DataFrame, float]:
    """
    Long-only backtest for one symbol with:
      - ATR-based stop and target
      - Risk-based position sizing + max position cap
      - Trend-based exit (environment change)
      - Optional max holding days as a safety net

    Assumes df has columns:
      - open, high, low, close
      - ema_20, ema_50
      - atr_14
      - long_signal (bool)
    """

    df = df.copy().sort_index()

    equity = initial_equity
    trades = []

    in_position = False
    entry_price = 0.0
    entry_date = None
    shares = 0
    stop_price = 0.0
    target_price = 0.0
    bars_held = 0
    risk_dollars_actual = 0.0  # actual risk used for this trade

    for i in range(len(df) - 1):
        row = df.iloc[i]
        next_row = df.iloc[i + 1]
        date = df.index[i]
        next_date = df.index[i + 1]

        if not in_position:
            # ENTRY LOGIC: check today's bar; enter on next day's open
            if bool(row.get("long_signal", False)):
                atr = float(row["atr_14"])
                stop_dist = strategy_settings.atr_multiple_stop * atr

                if stop_dist <= 0:
                    continue

                entry_price_candidate = float(next_row["open"])

                # 1) Risk-based position sizing
                desired_risk_dollars = equity * strategy_settings.risk_per_trade
                shares_risk = int(desired_risk_dollars // stop_dist)

                # 2) Position-size cap
                max_position_value = equity * strategy_settings.max_position_pct
                shares_cap = int(max_position_value // entry_price_candidate)

                shares = min(shares_risk, shares_cap)

                if shares < 1:
                    continue

                entry_price = entry_price_candidate
                entry_date = next_date

                stop_price = entry_price - stop_dist
                target_price = entry_price + strategy_settings.atr_multiple_target * atr

                risk_dollars_actual = shares * stop_dist

                in_position = True
                bars_held = 0

        else:
            # POSITION MANAGEMENT using next day's bar
            bars_held += 1

            low = float(next_row["low"])
            high = float(next_row["high"])
            close = float(next_row["close"])

            # We need today's trend context too
            ema20 = float(next_row.get("ema_20", close))
            ema50 = float(next_row.get("ema_50", close))

            exit_price = None
            exit_reason = None

            # 1) Stop-loss (assume worse-case ordering)
            if low <= stop_price:
                exit_price = stop_price
                exit_reason = "stop"

            # 2) Target
            elif high >= target_price:
                exit_price = target_price
                exit_reason = "target"

            else:
                # 3) Trend-based exit (environment change)
                trend_broken = (close < ema20) or (ema20 < ema50)

                if trend_broken:
                    exit_price = close
                    exit_reason = "trend"

                # 4) Time-based exit as a last resort
                elif bars_held >= strategy_settings.max_holding_days:
                    exit_price = close
                    exit_reason = "time"

            if exit_price is not None:
                exit_date = next_date
                pnl = (exit_price - entry_price) * shares
                position_value = entry_price * shares
                ret_pct = pnl / position_value if position_value > 0 else 0.0
                R = pnl / risk_dollars_actual if risk_dollars_actual > 0 else 0.0

                equity += pnl

                trades.append(
                    {
                        "symbol": symbol,
                        "entry_date": entry_date,
                        "entry_price": entry_price,
                        "exit_date": exit_date,
                        "exit_price": exit_price,
                        "shares": shares,
                        "risk_dollars": risk_dollars_actual,
                        "pnl": pnl,
                        "R": R,
                        "return_pct": ret_pct,
                        "exit_reason": exit_reason,
                        "equity_after": equity,
                    }
                )

                # Reset position state
                in_position = False
                entry_price = 0.0
                entry_date = None
                shares = 0
                stop_price = 0.0
                target_price = 0.0
                bars_held = 0
                risk_dollars_actual = 0.0

    trades_df = pd.DataFrame(trades)
    return trades_df, equity
