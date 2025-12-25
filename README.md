# Project Purple — Momentum Swing Trading System (Long-Only)

Project Purple is a **Python swing-trading research + paper-trading project** focused on **long-only momentum / trend-following** using **daily bars**.

The current codebase is primarily a **research harness**:
- build a tradeable universe
- generate long signals
- backtest with basic realism (ATR stops/targets, time-stop, slippage)
- apply a simple “one-position-at-a-time” portfolio policy

Execution to Interactive Brokers (paper) is **planned**, but **not yet wired for live order placement**.

---

## What it does today (current state as of December 24, 2025)

When you run the main entrypoint, Project Purple:

1) **Determines market regime (SPY)** using:
   - `BULL`: close > 200MA **and** 50MA > 200MA  
   - `BEAR`: close < 200MA  
   - `CHOPPY`: everything else  
   (`CHOPPY` applies a reduced risk multiplier; `BEAR` disables longs)

2) **Builds a candidate universe** (fast screen + optional “slow” filter):
   - liquidity / price checks
   - simple momentum / volatility scoring
   - optional per-symbol system backtest as an “edge-response” filter

3) **Backtests each symbol** with:
   - entry on signal transition (0→1)
   - ATR-based stop + target
   - exits: stop / target / signal-exit / time-exit (default 10 bars)
   - slippage applied on entry and exit (constant %)

4) **Applies Portfolio Policy A** (global combination across symbols):
   - choose **best trade per day** by `entry_score` (if present)
   - enforce **one position at a time** (no overlapping trades)
   - compute equity curve + total/avg R + max drawdown

---

## Canonical data location (IMPORTANT)

**All market data CSVs live in the repo-root `./data/` folder (Option A).**

✅ Correct:
- `./data/SPY_daily.csv`
- `./data/AAPL_daily.csv`
- `./data/<SYMBOL>_daily.csv`

🚫 Not used (and should not exist):
- `./project_purple/data/`

This project previously suffered from confusing duplicate data folders. The codebase is being locked to **repo-root `./data/` only** to prevent corrupted/shifted OHLCV data and path ambiguity.

---

## Repository layout (important files)

Top-level:
- `run_project_purple.py` — **recommended launcher** (works around current import style)
- `requirements.txt` — python deps
- `.env` — IB connection defaults (host/port/client_id) *(not tracked)*
- `data/` — **canonical daily OHLCV CSVs** (`*_daily.csv`)
- `project_purple/` — Python package

Package:
- `project_purple/main.py` — orchestrates regime → universe → backtests → portfolio policy
- `project_purple/scanner_simple.py` — universe selection + scoring
- `project_purple/backtest_v2.py` — backtest engine + trade simulation
- `project_purple/risk.py` — ATR stop/target + position sizing + R-multiple
- `project_purple/market_state.py` — regime classification (SPY)
- `project_purple/trade_plan.py` — prints a “today trade plan” style report (no order placement)
- `project_purple/data_loader.py` — loads per-symbol daily CSVs from **repo-root `./data/`**
- `project_purple/data_downloader.py` — downloads daily bars via yfinance into **repo-root `./data/`**
- `project_purple/ib_client.py` — IBKR historical data loader (ib_insync)
- `project_purple/archive/` — older experiments (not used by the main run)

Data (canonical):
- `data/*_daily.csv` — daily OHLCV files used by the loader

---

## Quickstart (Windows / macOS / Linux)

### 1) Create a virtual environment
From the repo root (the folder containing `run_project_purple.py`):

```bash
python -m venv .venv
```

Activate it:
- **Windows (PowerShell)**:
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
- **macOS / Linux**:
  ```bash
  source .venv/bin/activate
  ```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Ensure you have data (repo-root `./data`)
You need at minimum:
- `data/SPY_daily.csv`

Optional: download/refresh data via yfinance using the downloader.

**Safe default (does NOT overwrite existing `*_daily.csv`):**
```powershell
python -u .\project_purple\data_downloader.py --symbols SPY
```

If `data/SPY_daily.csv` already exists, the downloader will write a timestamped file like:
- `data/SPY_daily.download_YYYYMMDD_HHMMSS.csv`

**Overwrite existing files (use with care):**
```powershell
python -u .\project_purple\data_downloader.py --symbols SPY --overwrite
```

### 4) Run the system (recommended)
```bash
python run_project_purple.py
```

You should see:
- market regime printout (SPY)
- chosen universe size
- backtest summary (trades, total R, final equity, max drawdown)

---

## Notes on the current run style (why the launcher exists)

Some modules use **local imports** (e.g., `from risk import RiskConfig`) which is convenient during early prototyping but fragile when running as a package.

`run_project_purple.py` sets the working directory and `sys.path` so the current structure runs consistently.

**Roadmap item:** refactor imports to strict package imports (e.g., `from project_purple.risk import ...`) and remove the launcher.

---

## Key trading concepts used here (plain English)

- **Momentum / Trend following:** buy stocks already moving up (with filters to avoid junk).
- **ATR (Average True Range):** a volatility measure. Bigger ATR = wider daily swings.
- **ATR stop / target:** stop is `entry - k*ATR`, target is `entry + m*ATR`.
- **R-multiple (“R”):** profit/loss expressed in units of your initial risk.
  - +1R means you made what you risked; -1R means you lost what you risked.
- **Slippage:** a small penalty that approximates imperfect fills (modeled as a fixed % on entry/exit).

---

## What’s missing before IB paper trading (honest status)

You have **IB connectivity for historical bars** (`ib_client.py`), but **you do not yet have** a production-safe execution layer.

Before paper trading, you still need:
- an **execution module** to place bracket/OCO orders (entry + stop + target)
- order/position **reconciliation** (avoid duplicate orders, handle partial fills, reconnects)
- a **trade journal/log** (CSV/JSON) with order IDs and daily decisions
- safety gates (max trades/day, max exposure, “already in position” checks)

---

## How we should use Codex to keep the code clean (recommended workflow)

Codex is best used as a repo-aware coding agent to propose small, reviewable diffs.

**Rule for this repo:** _one change, one file, one test._

Suggested workflow:
1) Write a tiny task spec (what + why + acceptance test).
2) Tell Codex: “Change **only one file** and run **one** test command.”
3) Review the diff, then merge.

High-impact Codex tasks for this repo:
- **Import cleanup:** convert local imports → `project_purple.*` package imports (remove the launcher).
- **Prevent lookahead:** add tests to verify signals/entries don’t use future data.
- **Execution module:** implement IB paper bracket orders behind a `--dry-run` flag.
- **Logging:** add a structured daily log (decisions + orders + fills).
- **Linting/typing:** add Ruff + MyPy and fix the easy wins.

---

## Safety / Disclaimer

This project is for educational and research purposes.
Paper trade first. Expect bugs.
Markets are risky, and results can differ substantially from backtests.
