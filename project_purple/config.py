# project_purple/config.py

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


# Base directory of the project (top-level project_purple folder)
BASE_DIR = Path(__file__).resolve().parents[1]

# Load environment variables from .env in the project root
load_dotenv(BASE_DIR / ".env")


@dataclass
class AppConfig:
    """
    Application configuration for Project Purple.
    Anything related to environment, paths, and external services
    (like Interactive Brokers) lives here.
    """

    # Environment (dev, backtest, live, etc.)
    env: str = os.getenv("ENV", "dev")

    # Paths
    data_dir: Path = BASE_DIR / "data"
    docs_dir: Path = BASE_DIR / "docs"
    log_dir: Path = BASE_DIR / "logs"

    # Interactive Brokers connection settings
    ib_host: str = os.getenv("IB_HOST", "127.0.0.1")
    ib_port: int = int(os.getenv("IB_PORT", "7497"))
    ib_client_id: int = int(os.getenv("IB_CLIENT_ID", "1"))


# Single global config instance the rest of the app can import
config = AppConfig()
