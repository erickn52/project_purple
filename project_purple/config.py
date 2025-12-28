# project_purple/config.py

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

from dotenv import load_dotenv


# Base directory of the project (repo root)
BASE_DIR = Path(__file__).resolve().parents[1]

# Load environment variables from .env in the project root (optional)
load_dotenv(BASE_DIR / ".env")


def _env_str(name: str, default: str) -> str:
    """Read a string env var safely (never throws at import-time)."""
    val = os.getenv(name)
    if val is None:
        return default
    val = val.strip()
    return val if val else default


def _env_float(name: str, default: float) -> float:
    """Read a float env var safely (never throws at import-time)."""
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


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


# ---- Strategy defaults (P1 config centralization) ---------------------------------
#
# IMPORTANT:
# - This module must be safe to import from anywhere (no heavy imports, no IO).
# - Values here are "single source of truth" defaults for the live daily plan.
# - Environment variables can override some values for quick experiments without code edits.

DEFAULT_CANDIDATE_SYMBOLS: Tuple[str, ...] = (
    "AAPL", "AMD", "AMDL", "AMZN", "IBKR",
    "META", "MSFT", "NVDA", "SPY", "TSLA",

    "PLTR", "SHOP", "UBER", "SNAP", "PINS",
    "ZM", "CRWD", "NET", "PATH",
    "OKTA", "HUBS", "DDOG",

    "DKNG", "CROX", "LULU", "ETSY", "RBLX",
    "CELH", "PTON", "ROKU", "FVRR", "LYFT",

    "RIVN", "LCID", "RUN", "ENPH", "FSLR",

    "SOFI", "COIN", "HOOD",

    "NVAX", "BNTX", "MRNA", "IONS", "REGN",

    "BA", "GE", "UAL", "DAL",

    "WBD", "DIS",

    "OXY", "APA", "FCX", "AA",

    "AFRM", "MDB", "ZS", "TEAM",
)


@dataclass(frozen=True)
class StrategyConfig:
    """Defaults for the daily trade-plan workflow."""

    # Market regime benchmark ticker (used by market_state.py)
    benchmark_ticker: str = field(
        default_factory=lambda: _env_str("PP_BENCHMARK_TICKER", "SPY")
    )

    # Candidate universe for scanning/ranking (used by scanner/downloader)
    candidate_symbols: Tuple[str, ...] = DEFAULT_CANDIDATE_SYMBOLS

    # Default equity used by the launcher for ticket recomputation (gap checks)
    default_equity: float = field(
        default_factory=lambda: _env_float("PP_DEFAULT_EQUITY", 100_000.0)
    )


@dataclass(frozen=True)
class GapGuardConfig:
    """
    Defaults for the premarket/open gap-risk guard.

    These match the current launcher logic:
    - GAP UP: reject if open > min(entry*(1+max_gap_pct), entry + max_gap_atr*ATR)
    - GAP DOWN: reject if open <= stop OR (open - stop) < gap_down_too_tight_atr_mult*ATR
    """

    max_gap_pct: float = field(default_factory=lambda: _env_float("PP_GAP_MAX_PCT", 0.03))
    max_gap_atr: float = field(default_factory=lambda: _env_float("PP_GAP_MAX_ATR", 0.50))
    gap_down_too_tight_atr_mult: float = field(
        default_factory=lambda: _env_float("PP_GAP_DOWN_TOO_TIGHT_ATR_MULT", 0.33)
    )


# Single global config instances the rest of the app can import
config = AppConfig()
strategy_config = StrategyConfig()
gap_guard_config = GapGuardConfig()
