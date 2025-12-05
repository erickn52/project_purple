from typing import Tuple

import pandas as pd

from project_purple.settings import strategy_settings


def run_simple_backtest(
    df: pd.DataFrame,
    symbol: str,
    initial_equity: float = 100_000.0,
) -> Tuple[pd.DataFrame, float]:
    """
    Very simple long-only backtest for one symbol.

    Assumes df has columns:
      - 'open', 'high', 'low', 'close'
      - 'atr_14'
      - 'long_signal' (bool)
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
    risk_dollars = 0.0  # risk for the current trade

    for i in range(len(df) - 1):
        row = df.iloc[i]
        next_row = df.iloc[i + 1]
        date = df.index[i]
        next_date = df.index[i + 1]

        if not in_position:
            if bool(row.get("long_signal", False)):
                atr = float(row["atr_14"])
                stop_dist = strategy_settings.atr_multiple_stop * atr

                if stop_dist <= 0:
                    continue

                # Position sizing by risk
                risk_dollars = equity * strategy_settings.risk_per_trade
                shares = int(risk_dollars // stop_dist)

                if shares < 1:
                    continue

                entry_price = float(next_row["open"])
                entry_date = next_date

                stop_price = entry_price - stop_dist
                target_price = entry_price + strategy_settings.atr_multiple_target * atr

                in_position = True
                bars_held = 0

        else:
            bars_held += 1

            low = float(next_row["low"])
            high = float(next_row["high"])
            close = float(next_row["close"])

            exit_price = None
            exit_reason = None

            if low <= stop_price:
                exit_price = stop_price
                exit_reason = "stop"
            elif high >= target_price:
                exit_price = target_price
                exit_reason = "target"
            elif bars_held >= strategy_settings.max_holding_days:
                exit_price = close
                exit_reason = "time"

            if exit_price is not None:
                exit_date = next_date
                pnl = (exit_price - entry_price) * shares
                ret_pct = pnl / (entry_price * shares) if shares > 0 else 0.0
                R = pnl / risk_dollars if risk_dollars > 0 else 0.0

                equity += pnl

                trades.append(
                    {
                        "symbol": symbol,
                        "entry_date": entry_date,
                        "entry_price": entry_price,
                        "exit_date": exit_date,
                        "exit_price": exit_price,
                        "shares": shares,
                        "risk_dollars": risk_dollars,
                        "pnl": pnl,
                        "R": R,
                        "return_pct": ret_pct,
                        "exit_reason": exit_reason,
                        "equity_after": equity,
                    }
                )

                in_position = False
                entry_price = 0.0
                shares = 0
                stop_price = 0.0
                target_price = 0.0
                bars_held = 0
                risk_dollars = 0.0

    trades_df = pd.DataFrame(trades)
    return trades_df, equity
