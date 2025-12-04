from project_purple import config, settings
from project_purple.ib_client import ib_client


def main():
    # 1) Show config & settings (sanity check)
    config.show_config_summary()

    print("\n=== STRATEGY SETTINGS ===")
    print(f"Price range: {settings.MIN_PRICE} - {settings.MAX_PRICE}")
    print(f"Minimum dollar volume: {settings.MIN_DOLLAR_VOLUME:,}")
    print(f"Stop loss: {settings.STOP_LOSS_PCT}%")
    print(f"Risk per trade: {settings.RISK_PER_TRADE * 100}%")
    print("==================================\n")

    # 2) Connect to IB
    connected = ib_client.connect()
    if not connected:
        print("Could not connect to IB. Make sure TWS/Gateway is running and API is enabled.")
        return

    # 3) Request server time as a simple, safe test call
    server_time = ib_client.get_server_time()
    if server_time is not None:
        print(f"IB server time: {server_time}")
    else:
        print("Failed to retrieve server time.")

    # 4) Disconnect cleanly
    ib_client.disconnect()


if __name__ == "__main__":
    main()
