# run_project_purple.py

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional


def _enable_windows_ansi() -> None:
    """Best-effort enabling of ANSI escape codes on Windows consoles."""
    if os.name != "nt":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE = -11
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
    except Exception:
        pass


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Project Purple daily plan end-to-end.")
    p.add_argument("--refresh-data", action="store_true", help="Refresh CSVs before building the plan.")
    p.add_argument(
        "--symbols",
        nargs="*",
        default=None,
        help="Optional symbols to refresh (e.g., --symbols SPY AAPL). If omitted, downloader decides default.",
    )
    p.add_argument("--period", default="10y", help="yfinance period (default: 10y)")
    p.add_argument("--interval", default="1d", help="yfinance interval (default: 1d)")
    p.add_argument("--overwrite", action="store_true", help="Pass --overwrite to the downloader if supported.")
    p.add_argument("--as-of", default=None, help="Optional historical as-of date (YYYY-MM-DD).")
    p.add_argument("--refresh-only", action="store_true", help="Only refresh data; do not build a trade plan.")
    p.add_argument("--no-color", action="store_true", help="Disable ANSI color output.")
    return p.parse_args()


def _run_downloader(repo_root: Path, pkg_dir: Path, args: argparse.Namespace) -> None:
    downloader_path = pkg_dir / "data_downloader.py"
    if not downloader_path.exists():
        raise FileNotFoundError(f"Expected file not found: {downloader_path}")

    cmd: List[str] = [sys.executable, "-u", str(downloader_path)]
    if args.symbols:
        cmd += ["--symbols", *args.symbols]
    if args.period:
        cmd += ["--period", args.period]
    if args.interval:
        cmd += ["--interval", args.interval]
    if args.overwrite:
        cmd += ["--overwrite"]

    print("\n=== REFRESH DATA ===")
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(repo_root), check=True)


def _fmt_price(x: Any) -> str:
    try:
        return f"{float(x):.2f}"
    except Exception:
        return ""


def _fmt_money(x: Any) -> str:
    try:
        return f"{float(x):,.2f}"
    except Exception:
        return ""


def _fmt_pct_decimal(x: Any) -> str:
    """x is decimal fraction (0.01 = 1%)."""
    try:
        return f"{float(x) * 100:.3f}%"
    except Exception:
        return ""


def _fmt_date(x: Any) -> str:
    try:
        import pandas as pd
        ts = pd.to_datetime(x)
        return ts.strftime("%Y-%m-%d")
    except Exception:
        return str(x) if x is not None else ""


def _colors_enabled() -> bool:
    return not bool(os.environ.get("NO_COLOR"))


def _color(text: str, code: str) -> str:
    if not _colors_enabled():
        return text
    return f"{code}{text}\033[0m"


def _color_regime(regime: str) -> str:
    if regime == "BULL":
        return _color(regime, "\033[92m")  # green
    if regime == "BEAR":
        return _color(regime, "\033[91m")  # red
    return _color(regime, "\033[93m")      # yellow (CHOP/other)


def _color_bool(val: bool) -> str:
    return _color(str(val), "\033[92m") if val else _color(str(val), "\033[91m")


def _print_run_banner(repo_root: Path, args: argparse.Namespace) -> None:
    local_dt = datetime.now().astimezone()
    utc_dt = datetime.now(timezone.utc)

    print("\n=== PROJECT PURPLE (DAILY PLAN) ===")
    print(f"Repo root:  {repo_root}")
    print(f"Python:     {sys.executable}")
    print(f"Started:    {local_dt.strftime('%Y-%m-%d %H:%M:%S %Z%z')} (local)")
    print(f"Started:    {utc_dt.strftime('%Y-%m-%d %H:%M:%S UTC')} (UTC)")
    print(f"As-of arg:  {args.as_of if args.as_of else '(none)'}")


def _print_market_state(result: dict) -> None:
    print("\n=== MARKET STATE ===")
    regime = str(result.get("regime", ""))
    trade_long = bool(result.get("trade_long_allowed", False))
    risk_mult = result.get("risk_multiplier", 0.0)

    print(f"Regime:     {_color_regime(regime)}")
    print(f"Trade long: {_color_bool(trade_long)}")
    try:
        print(f"Risk mult:  {float(risk_mult):.2f}x")
    except Exception:
        print(f"Risk mult:  {risk_mult}")

    if result.get("status") == "blocked_policy_error":
        err = result.get("policy_error", "")
        if err:
            print(_color("Policy error:", "\033[91m"), err)


def _print_trade_plan(result: dict) -> None:
    status = result.get("status")

    if status != "planned":
        print("\n=== RESULT ===")
        print(f"Status: {status}")
        note = result.get("note", "")
        if note:
            print(f"Note:   {note}")
        return

    rc = result.get("risk_config")

    print("\n=== TRADE PLAN (Policy A) ===")
    print(f"{'symbol':>18}: {result.get('symbol', '')}")
    print(f"{'as_of_date':>18}: {_fmt_date(result.get('as_of_date'))}")
    print(f"{'regime':>18}: {result.get('regime', '')}")
    print(f"{'risk_multiplier':>18}: {result.get('risk_multiplier', '')}")

    if rc is not None:
        print(f"{'risk_config':>18}:")
        print(f"{'':>18}  risk_per_trade_pct      = {_fmt_pct_decimal(getattr(rc, 'risk_per_trade_pct', None))}")
        print(f"{'':>18}  atr_stop_multiple       = {getattr(rc, 'atr_stop_multiple', '')}")
        print(f"{'':>18}  atr_target_multiple     = {getattr(rc, 'atr_target_multiple', '')}")
        print(f"{'':>18}  max_pos_pct_of_equity   = {_fmt_pct_decimal(getattr(rc, 'max_position_pct_of_equity', None))}")
        print(f"{'':>18}  min_shares              = {getattr(rc, 'min_shares', '')}")

    print(f"{'entry_price':>18}: {_fmt_price(result.get('entry_price'))}")
    print(f"{'stop_price':>18}: {_fmt_price(result.get('stop_price'))}")
    print(f"{'target_price':>18}: {_fmt_price(result.get('target_price'))}")

    shares = result.get("shares")
    try:
        shares_disp = str(int(float(shares))) if shares is not None else ""
    except Exception:
        shares_disp = str(shares) if shares is not None else ""

    print(f"{'shares':>18}: {shares_disp}")
    print(f"{'dollars_at_risk':>18}: {_fmt_money(result.get('dollars_at_risk'))}")


def main() -> None:
    args = _parse_args()

    repo_root = Path(__file__).resolve().parent
    pkg_dir = repo_root / "project_purple"
    if not pkg_dir.exists():
        raise FileNotFoundError(f"Expected folder not found: {pkg_dir}")

    # Canonical working directory for ./data and ./logs
    os.chdir(repo_root)

    if args.no_color:
        os.environ["NO_COLOR"] = "1"
    else:
        _enable_windows_ansi()

    _print_run_banner(repo_root=repo_root, args=args)

    if args.refresh_data:
        _run_downloader(repo_root=repo_root, pkg_dir=pkg_dir, args=args)

    if args.refresh_only:
        print("\nRefresh-only requested. Exiting without building the trade plan.")
        return

    from project_purple.trade_plan import build_trade_plan

    result = build_trade_plan(as_of_date=args.as_of, return_meta=True)

    # result is always a dict here
    _print_market_state(result)
    _print_trade_plan(result)


if __name__ == "__main__":
    main()
