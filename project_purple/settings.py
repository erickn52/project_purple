"""
settings.py
Strategy-level defaults and trading parameters.
These do NOT belong in .env (those are for secrets & environment-specific values).
"""

# ----------------------------
# PRICE FILTERS
# ----------------------------
MIN_PRICE = 12.0
MAX_PRICE = 60.0

# ----------------------------
# VOLUME FILTERS
# ----------------------------
# Minimum average daily dollar volume (liquidity filter)
MIN_DOLLAR_VOLUME = 2_000_000     # $2M

# ----------------------------
# MOMENTUM FILTERS
# ----------------------------
# % gain over X days for momentum screen
MOMENTUM_LOOKBACK_DAYS = 5
MIN_MOMENTUM_PCT = 4.0    # require at least +4% move

# ----------------------------
# RISK MANAGEMENT
# ----------------------------
# Stop-loss distances
STOP_LOSS_PCT = 2.0        # 2% default risk
TRAILING_STOP_PCT = 1.5    # optional trailing stop

# ----------------------------
# POSITION SIZING
# ----------------------------
RISK_PER_TRADE = 0.01      # 1% of account per trade

# ----------------------------
# GENERAL SETTINGS
# ----------------------------
VERBOSE = True             # print debug information
LOG_TRADES = True          # write trades to logs (later)
