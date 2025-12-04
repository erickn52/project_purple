from pathlib import Path
import os

from dotenv import load_dotenv

# Base directory of the project (top level folder)
BASE_DIR = Path(__file__).resolve().parent.parent

# Path to the .env file
ENV_PATH = BASE_DIR / ".env"

# Load environment variables from .env if it exists
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    # Optional: helpful warning if .env is missing
    print(f"WARNING: .env file not found at {ENV_PATH}. Using default values where possible.")

# --- Interactive Brokers settings ---

IB_HOST = os.getenv("IB_HOST", "127.0.0.1")
IB_PORT = int(os.getenv("IB_PORT", "7497"))
IB_CLIENT_ID = int(os.getenv("IB_CLIENT_ID", "1"))

# --- Paths ---

DATA_PATH = BASE_DIR / os.getenv("DATA_PATH", "data")

# Ensure data directory exists
DATA_PATH.mkdir(parents=True, exist_ok=True)

# --- Strategy defaults (we'll actually use these later) ---

MIN_PRICE = float(os.getenv("MIN_PRICE", "12"))
MAX_PRICE = float(os.getenv("MAX_PRICE", "60"))
MIN_AVG_DOLLAR_VOLUME = float(os.getenv("MIN_AVG_DOLLAR_VOLUME", "2000000"))


def show_config_summary() -> None:
    """
    Simple helper to print the current configuration.
    Useful for debugging and sanity checks.
    """
    print("=== Swing Trader Configuration ===")
    print(f"BASE_DIR:          {BASE_DIR}")
    print(f"ENV_PATH:          {ENV_PATH}")
    print(f"IB_HOST:           {IB_HOST}")
    print(f"IB_PORT:           {IB_PORT}")
    print(f"IB_CLIENT_ID:      {IB_CLIENT_ID}")
    print(f"DATA_PATH:         {DATA_PATH}")
    print(f"MIN_PRICE:         {MIN_PRICE}")
    print(f"MAX_PRICE:         {MAX_PRICE}")
    print(f"MIN_$VOLUME:       {MIN_AVG_DOLLAR_VOLUME}")
    print("==================================")
