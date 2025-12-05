# project_purple/main.py

from __future__ import annotations

from project_purple.data_loader import load_history
from project_purple.indicators import add_basic_trend_indicators
from project_purple.signals import add_basic_long_signal


def main() -> None:
    for symbol in ["AAPL", "NVDA"]:
        # 1) Load history
        df = load_history(symbol, source="csv")

        # 2) Add indicators (EMAs, ATR, 20d high)
        df = add_basic_trend_indicators(df)

        # 3) Add long-entry signal
        df = add_basic_long_signal(df)

        print(f"{symbol} data loaded: {len(df)} rows")

        # Show last 10 rows with the signal
        cols_to_show = [
            "symbol",
            "close",
            "ema_20",
            "ema_50",
            "atr_14",
            "high_20d",
            "long_signal",
        ]
        print(df[cols_to_show].tail(10), "\n")


if __name__ == "__main__":
    main()
