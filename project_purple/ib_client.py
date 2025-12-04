"""
ib_client.py
Thin wrapper around ib_insync.IB for connecting to Interactive Brokers.

This module uses the config values defined in config.py and exposes a
single shared IBClient instance (`ib_client`) for the rest of the app.

Safety:
- Use TWS/Gateway PAPER account.
- Keep API in read-only mode while testing.
"""

from datetime import datetime
from typing import Optional

from ib_insync import IB  # type: ignore

from . import config


class IBClient:
    def __init__(self) -> None:
        self.ib = IB()

    # ------------------------
    # CONNECTION MANAGEMENT
    # ------------------------
    def connect(self) -> bool:
        """
        Connect to IB using host/port/clientId from config.py.
        Returns True if connected, False otherwise.
        """
        if self.ib.isConnected():
            print("IB: Already connected.")
            return True

        print(
            f"IB: Connecting to {config.IB_HOST}:{config.IB_PORT} "
            f"with clientId={config.IB_CLIENT_ID}..."
        )

        try:
            self.ib.connect(
                host=config.IB_HOST,
                port=config.IB_PORT,
                clientId=config.IB_CLIENT_ID,
                readonly=True,  # extra safety layer
            )
        except Exception as exc:
            print(f"IB: Connection error -> {exc}")
            return False

        if self.ib.isConnected():
            print("IB: Connected successfully.")
            return True

        print("IB: Failed to connect (unknown reason).")
        return False

    def disconnect(self) -> None:
        """Disconnect from IB if currently connected."""
        if self.ib.isConnected():
            self.ib.disconnect()
            print("IB: Disconnected.")
        else:
            print("IB: Not connected; nothing to disconnect.")

    def is_connected(self) -> bool:
        """Return True if connected to IB, else False."""
        return self.ib.isConnected()

    # ------------------------
    # SIMPLE TEST HELPERS
    # ------------------------
    def get_server_time(self) -> Optional[datetime]:
        """
        Return the IB server time if connected, otherwise None.
        This is a safe way to confirm the API is working.
        """
        if not self.ib.isConnected():
            print("IB: Not connected; cannot get server time.")
            return None

        try:
            server_time = self.ib.reqCurrentTime()
            return server_time
        except Exception as exc:
            print(f"IB: Error requesting server time -> {exc}")
            return None


# A single shared instance for convenience:
ib_client = IBClient()
