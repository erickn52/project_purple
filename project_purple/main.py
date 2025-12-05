from project_purple import config, settings
from project_purple.ib_client import ib_client
from project_purple.data_loader import get_daily_bars


def main():
    # 1) Show config & settings
    config.show_config_summary()

    print("\n=== STRATEGY SETTINGS ===")
    print(f"Price range: {settings.MIN_PRICE} - {settings.MAX_PRICE}")
    print(f"Minimum dollar volume: {settings.MIN_DOLLAR_VOLUME:,}")
    print(f"Stop loss: {settings.STOP_LOSS_PCT}%")
    print(f"Risk per trade: {settings.RISK_PER_TRADE * 100}%")
    print("==================================\n")

    # 2) Connect to IB
    if not ib_client.connect():
        print("Could not connect to IB. Make sure TWS/Gateway is running and API is enabled.")
        return

    # 3) Request some historical data as a test
    test_symbol = "NVDA"   # you can change this to any liquid stock you like
    lookback_days = 60

    df = get_daily_bars(symbol=test_symbol, lookback_days=lookback_days, save_csv=True)

    if df is not None:
        print(f"\nLoaded {len(df)} rows of daily data for {test_symbol}.")
        print(df.tail(5))  # show the last few rows
    else:
        print("No data returned.")

    # 4) Disconnect cleanly
    ib_client.disconnect()


if __name__ == "__main__":
    main()
