# project_purple/main.py

from __future__ import annotations

from project_purple.data_loader import load_history


def main() -> None:
    # Test loading from CSV for our sample symbols
    for symbol in ["AAPL", "NVDA"]:
        df = load_history(symbol, source="csv")
        print(f"{symbol} data loaded: {len(df)} rows")
        print(df.head(), "\n")


if __name__ == "__main__":
    main()
