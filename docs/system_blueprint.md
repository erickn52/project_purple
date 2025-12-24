# Project Purple — System Blueprint (Canonical v1)

This document is the single source for what Project Purple **actually does today** when run via the canonical entrypoint.

---

## Entrypoints 

### Canonical entrypoint (system run)
**Command:**
- `python run_project_purple.py`

**What it does:**
- `run_project_purple.py` changes the working directory to the package folder and adjusts `sys.path`,
  then executes `project_purple/main.py` as `__main__`.
- This exists because the project currently uses local-style imports like `from risk import RiskConfig`,
  which depend on being executed from inside the package folder.

### Secondary entrypoint (component-only debug run)
**Command:**
- `python project_purple/backtest_v2.py`

**What it does:**
- Runs `backtest_v2.py`’s internal `main()` which backtests a single symbol (default `"AAPL"`).
- This is **not** the canonical system path; it is a standalone debug runner.

---

## End-to-end execution flow (canonical system run)

When you run `python run_project_purple.py`, the system executes this flow:

### Step 0 — Launch main orchestrator
- `run_project_purple.py` executes: `project_purple/main.py`

### Step 1 — Market regime (benchmark = "SPY")
File: `project_purple/market_state.py`  
Called from: `project_purple/main.py`

Regime logic (exact):
- Compute MA50 and MA200 on SPY.
- **BULL** if: `close > MA200` AND `MA50 > MA200`
- **BEAR** if: `close < MA200`
- **CHOPPY** otherwise

Risk multipliers (exact):
- BULL  = 1.00
- CHOPPY = 0.50
- BEAR  = 0.00

Output printed by main:
- Regime, TradeLong, Risk multiplier

### Step 2 — Risk configuration (built in main.py)
File: `project_purple/main.py`

Exact parameters:
- `base_risk_per_trade_pct = 0.01` (1%)
- `risk_per_trade_pct = base_risk_per_trade_pct * state.risk_multiplier`

RiskConfig is created with:
- `risk_per_trade_pct = (0.01 * regime_multiplier)`
- `atr_stop_multiple = 1.5`
- `atr_target_multiple = 3.0`
- `max_position_pct_of_equity = 0.20` (20%)
- `min_shares = 1`

### Step 3 — Universe selection
File: `project_purple/main.py`

Exact behavior:
- main tries to import and call `scanner_simple.build_universe()`.
- If unavailable or it returns nothing, main uses this fallback list:

`["AAPL","AMD","AMDL","AMZN","IBKR","META","MSFT","NVDA","RUN","SPY","TECL","TSLA"]`

### Step 4 — Per-symbol backtest loop (calls backtest_v2)
Files:
- Orchestrator: `project_purple/main.py`
- Component backtest: `project_purple/backtest_v2.py`

Exact parameters from main:
- `initial_equity = 100_000.0`
- `max_hold_days = 10`
- `trail_atr_multiple = 3.0`

For each symbol in the universe, main calls:
- `run_backtest_for_symbol(symbol=..., project_root=..., risk_config=..., initial_equity=100000, max_hold_days=10, trail_atr_multiple=3.0)`

Only non-empty trade DataFrames are collected into `all_trades`.

If no symbol produces trades:
- main prints: "No trades generated across universe."
- main returns early.

### Step 5 — Combined analysis (reporting)
File: `project_purple/backtest_v2.py`

Exact behavior:
- main concatenates all trades into `combined`
- calls `analyze_combined_trades(combined)`
- prints:
  - per-symbol stats (trades, win rate, avg R, total R, max drawdown)
  - global R distribution histogram buckets

### Step 6 — Portfolio Policy A (one position at a time)
File: `project_purple/main.py` (function `apply_portfolio_policy_A`)

Exact behavior:
1) Convert `entry_date` and `exit_date` to datetime
2) For each `entry_date`, select the best trade:
   - If `entry_score` exists: choose the highest `entry_score` for that day
   - If no `entry_score`: keep sorted by entry_date and take first available
3) Enforce no overlap:
   - Track `next_free_date` (initially very early)
   - Only take a trade if `entry_date >= next_free_date`
   - When a trade is taken, set `next_free_date = exit_date`
4) Update equity using R and fixed risk fraction (multiplicative):
   - `equity = equity * (1.0 + R * risk_per_trade_pct)`
   - where `risk_per_trade_pct` is the same value used in RiskConfig

Outputs printed:
- trades taken, total R, avg R, final equity, max drawdown %

---

## Canonical Strategy v1 (EXACT per-symbol logic in backtest_v2.py)

File: `project_purple/backtest_v2.py`

### Data requirements + cleaning (exact)
Input CSV must include:
- `date`, `symbol`, `open`, `high`, `low`, `close`, `volume`

Cleaning (`clean_aapl_dataframe`):
- If `Price` exists and `date` does not: rename `Price` → `date`
- Remove rows where `date` token is one of: `ticker`, `date`, `""`, `nan`, `none` (case-insensitive)
- Coerce `open/high/low/close/volume` to numeric
- Parse `date` to datetime (supports mixed formats; falls back to Excel serial-date parsing if needed)
- Drop rows missing any of: `date/open/high/low/close`
- Sort ascending by `date`

### Indicators (exact)
ATR14 (`add_atr`):
- True Range (TR) = max(
  - high - low,
  - abs(high - prev_close),
  - abs(low - prev_close)
)
- `atr = TR.rolling(14).mean()`

MA20/MA50 + signal (`add_momentum_pullback_signals`):
- `ma_fast = close.rolling(20).mean()`
- `ma_slow = close.rolling(50).mean()`

Signal condition (exact):
- Initialize `signal = 0`
- Set `signal = 1` when:
  - `close > ma_slow` AND `ma_fast > ma_slow`

Rows are then filtered to require non-null:
- `atr`, `ma_fast`, `ma_slow`

### Entry (exact)
Loop runs from index `i = 1` to end.

Enter long when:
- `prev_signal == 0` AND `signal == 1`

Slippage (exact):
- `SLIPPAGE_PCT = 0.001` (0.10%)

Entry price:
- `entry_price = close * (1.0 + SLIPPAGE_PCT)`

Risk sizing + initial stop/target:
- computed via `calculate_risk_for_trade(entry_price, atr, account_equity, config=risk_config)`
- if `shares <= 0`, skip the trade
- store:
  - `stop_price` and `initial_stop_price`
  - `target_price`
  - `shares`
  - `allowed_dollar_risk`
  - `account_equity_at_entry`

Entry score (exact; stored on trade):
If any input is non-finite or <= 0 → `-inf`
Otherwise:
- `trend1 = (close / ma_slow) - 1.0`
- `trend2 = (ma_fast / ma_slow) - 1.0`
- `vol_penalty = (atr / close)`
- `entry_score = (1.0*trend1) + (0.5*trend2) - (0.5*vol_penalty)`

### Trailing stop (exact; enabled by main.py)
Since main passes `trail_atr_multiple = 3.0`, trailing logic is active.

On each bar while in position:
- `candidate_stop = close - (trail_atr_multiple * atr_today)`
- if `candidate_stop > current stop_price`, update `stop_price = candidate_stop`

### Exits (exact priority order)
Let `hold_days = i - entry_index`.

Check exits in this order:
1) Stop:
   - if `low <= stop_price` → exit at `stop_price`, reason `"stop"`
2) Target:
   - else if `high >= target_price` → exit at `target_price`, reason `"target"`
3) Signal exit:
   - else if `prev_signal == 1` and `signal == 0` → exit at `close`, reason `"signal_exit"`
4) Time exit:
   - else if `hold_days >= max_hold_days` → exit at `close`, reason `"time_exit"`

Exit slippage (exact):
- `exit_price = exit_price * (1.0 - SLIPPAGE_PCT)`

R and equity update inside the component backtest:
- `R = calculate_R(exit_price, entry_price, initial_stop_price)`
- `pnl_dollars = (exit_price - entry_price) * shares`
- equity inside the symbol backtest is updated by `pnl_dollars`

---

## Non-canonical / not-in-main-path modules
Unless explicitly imported by `project_purple/main.py`, these are considered experimental:
- `signals.py`
- `indicators.py`

Also note: `project_purple/settings.py` defines strategy settings, but the canonical run path described above uses explicit parameters inside `main.py` and `RiskConfig` (not `StrategySettings`).
