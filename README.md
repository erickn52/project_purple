Project Purple — Swing Trading System
Overview

Project Purple is a professional-grade, Python-based swing trading system designed to:

Identify high-probability long setups using price action, trend, and volatility filters

Backtest signals with realistic stop-loss and take-profit behavior

Scale position sizes as equity grows

Run locally against Interactive Brokers for future automation

This repository contains:

Data loading (CSV + Interactive Brokers via ib_insync)

Technical indicators

Signal generation

Trade simulation & backtesting

A clean project structure for long-term model development

Project Structure
project_purple/
│
├── data/                     # Sample historical data (CSV)
│   ├── AAPL_daily_60d.csv
│   └── NVDA_daily_60d.csv
│
├── project_purple/
│   ├── __init__.py
│   ├── config.py             # Config settings (paths, IB settings, thresholds)
│   ├── data_loader.py        # Load CSV or IB historical data
│   ├── indicators.py         # EMA, ATR, rolling highs, etc.
│   ├── signals.py            # Long setup logic
│   ├── backtest.py           # Trade simulation engine
│   ├── ib_client.py          # IBKR interface
│   └── main.py               # Entry point / daily workflow
│
├── tests/                    # Future: unit tests
│
├── README.md                 # ← You are here
├── requirements.txt          # Frozen Python dependencies
├── .gitignore
└── .env                      # API keys + secrets (never commit)

1. Installation
Clone or download the project
git clone https://github.com/USERNAME/project_purple.git
cd project_purple

2. Create & activate a virtual environment
Windows
python -m venv venv
venv\Scripts\activate

Mac/Linux
python3 -m venv venv
source venv/bin/activate


You should see:

(venv)

3. Install dependencies

Install everything required for the model:

pip install -r requirements.txt

4. Add your .env file (required)

Create a .env in the project root:

IB_HOST=127.0.0.1
IB_PORT=7497
IB_CLIENT_ID=1


(If using IB Gateway or TWS, adjust port)

5. Running the System

Run:

python -m project_purple.main


This will:

Load AAPL + NVDA history

Calculate indicators

Generate signals

Backtest trades

Print summaries and final equity

Example output:

Summary for AAPL: 1 trades, win rate: 100%, avg R: 0.77
Final equity: 100,579.36

6. Development Workflow
To add new indicators

Edit:

project_purple/indicators.py

To adjust signal logic

Edit:

project_purple/signals.py

To tune risk rules

Edit:

project_purple/backtest.py

To change thresholds

Edit:

project_purple/settings.py

7. Future Modules (Planned)

Planned expansions for the system:

Volatility regime filters

Trend regime detection

Position sizing models (fixed fractional, Kelly-lite)

Scaling rules based on equity curve (your requested feature)

Walk-forward evaluation

Real-time execution via Interactive Brokers

8. License

Private use only.
Not intended for commercial distribution without permission.

9. Disclaimer

This software is for educational purposes only.
Trading involves risk.
Past performance does not guarantee future results.