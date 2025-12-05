# project_purple/ib_client.py

from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import pandas as pd
from ib_insync import IB, Stock, util  # type: ignore

from project_purple.config import config


class IBClient:
    """
    Thin wrapper around ib_insync.IB for Project Purple.
    Handles connectivity and raw historical data retrieval.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        client_id: Optional[int] = None,
    ) -> None:
        self.host = host or config.ib_host
        self.port = port or config.ib_port
        self.client_id = client_id or config.ib_client_id
        self.ib = IB()

    def connect(self) -> None:
        if not self.ib.isConnected():
            self.ib.connect(self.host, self.port, clientId=self.client_id)

    def disconnect(self) -> None:
        if self.ib.isConnected():
            self.ib.disconnect()

    @contextmanager
    def session(self):
        """
        Usage:
            with IBClient().session() as client:
                df = client.get_daily_history("AAPL")
        """
        self.connect()
        try:
            yield self
        finally:
            self.disconnect()

    def get_daily_history(
        self,
        symbol: str,
        duration: str = "60 D",
        bar_size: str = "1 day",
        what_to_show: str = "TRADES",
        use_rth: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch daily historical bars for a US stock from Interactive Brokers
        and return them as a standardized pandas DataFrame.
        """
        self.connect()

        contract = Stock(symbol, "SMART", "USD")
        bars = self.ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow=what_to_show,
            useRTH=use_rth,
            formatDate=1,
        )

        df = util.df(bars)

        if df.empty:
            return df

        # Standardize
        df.rename(
            columns={
                "date": "date",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
            },
            inplace=True,
        )

        df["symbol"] = symbol
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

        df = df[["symbol", "open", "high", "low", "close", "volume"]].sort_index()

        return df
