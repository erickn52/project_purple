# run_project_purple.py

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


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
            mode.value |= 0x0004  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
            kernel32.SetConsoleMode(handle, mode)
    except Exception:
        pass


def _colors_enabled() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return True


def _color(s: str, code: str) -> str:
    if not _colors_enabled():
        return s
    return f"{code}{s}\033[0m"


def _bold(s: str) -> str:
    """Best-effort bold styling (ANSI). Falls back to plain text when ANSI is disabled."""
    if not _colors_enabled():
        return s
    return f"\033[1m{s}\033[0m"


def _fmt_price(x: Optional[float]) -> str:
    if x is None:
        return ""
    return f"{x:.2f}"


def _fmt_money(x: Optional[float]) -> str:
    if x is None:
        return ""
    return f"{x:,.2f}"


def _fmt_pct_decimal(x: Optional[float]) -> str:
    """Format 0.01 -> 1.00%"""
    if x is None:
        return ""
    return f"{x*100.0:.2f}%"


def _repo_root() -> Path:
    # run_project_purple.py lives at repo root
    return Path(__file__).resolve().parent


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Project Purple daily plan end-to-end.")
    p.add_argument(
        "--refresh-data",
        action="store_true",
        help="Refresh data first by running project_purple/data_downloader.py",
    )
    p.add_argument(
        "--refresh-only",
        action="store_true",
        help="Only refresh data; do not build a trade plan.",
    )
    p.add_argument(
        "--as-of",
        default=None,
        help="Optional historical as-of date (YYYY-MM-DD).",
    )
    p.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI color output.",
    )

    # Gap-risk guard (P0 safety)
    p.add_argument(
        "--open",
        dest="open_price",
        default=None,
        help=(
            "Optional actual OPEN/FILL price you expect to enter at. "
            "If provided, Project Purple will evaluate gap-up OR gap-down risk and print a recomputed ticket."
        ),
    )
    p.add_argument(
        "--gap-max-pct",
        type=float,
        default=0.03,
        help="Max allowed gap above planned entry, as a decimal (0.03 = 3%). Default: 0.03.",
    )
    p.add_argument(
        "--gap-max-atr",
        type=float,
        default=0.50,
        help="Max allowed gap above planned entry, measured in ATR units (0.50 = half an ATR). Default: 0.50.",
    )

    return p.parse_args()


def _print_header(repo_root: Path, args: argparse.Namespace) -> None:
    _enable_windows_ansi()
    print("\n=== PROJECT PURPLE (DAILY PLAN) ===")
    print("Repo root: ", str(repo_root))
    print("Python:    ", sys.executable)

    now_local = datetime.now().astimezone()
    now_utc = datetime.now(timezone.utc)
    print("Started:   ", now_local.strftime("%Y-%m-%d %H:%M:%S %Z%z"), "(local)")
    print("Started:   ", now_utc.strftime("%Y-%m-%d %H:%M:%S UTC"), "(UTC)")
    print("As-of arg: ", args.as_of or "(none)")


def _run_data_refresh(repo_root: Path) -> int:
    print("\n=== DATA REFRESH ===")
    cmd = [sys.executable, str(repo_root / "project_purple" / "data_downloader.py")]
    print("Running:", " ".join(cmd))
    try:
        r = subprocess.run(cmd, cwd=str(repo_root), check=False)
        print("Exit code:", r.returncode)
        return int(r.returncode)
    except Exception as e:
        print("ERROR running data refresh:", str(e))
        return 1


def _print_market_state(result: dict) -> None:
    print("\n=== MARKET STATE ===")
    regime = result.get("regime", "UNKNOWN")
    trade_long = bool(result.get("trade_long_allowed", False))
    risk_mult = result.get("risk_multiplier", None)
    risk_mult_f = float(risk_mult) if risk_mult is not None else None

    # Color regime label
    regime_label = str(regime).upper()
    if regime_label == "BULL":
        regime_label = _color(regime_label, "\033[92m")  # green
    elif regime_label == "BEAR":
        regime_label = _color(regime_label, "\033[91m")  # red
    elif "CHOP" in regime_label:
        regime_label = _color(regime_label, "\033[93m")  # yellow
    else:
        regime_label = _color(regime_label, "\033[90m")  # gray

    print(f"{'Regime:':<12} {regime_label}")
    print(f"{'Trade long:':<12} {trade_long}")
    print(f"{'Risk mult:':<12} {'' if risk_mult_f is None else f'{risk_mult_f:.2f}x'}")


def _print_trade_plan(result: dict) -> None:
    print("\n=== RESULT ===")
    status = result.get("status", "unknown")
    note = result.get("note", "")
    print(f"{'Status:':<8} {status}")
    print(f"{'Note:':<8} {note}")

    if status != "planned":
        return

    print("\n=== TRADE PLAN ===")
    print(f"{'Symbol:':<14} {result.get('symbol', '')}")
    as_of_date = result.get("as_of_date")
    print(f"{'As-of date:':<14} {as_of_date}")

    entry = result.get("entry_price")
    stop = result.get("stop_price")
    target = result.get("target_price")
    shares = result.get("shares")
    risk_dollars = result.get("dollars_at_risk")

    entry_s = _fmt_price(float(entry)) if entry is not None else ""
    stop_s = _fmt_price(float(stop)) if stop is not None else ""
    target_s = _fmt_price(float(target)) if target is not None else ""

    # Bold the three key prices
    if entry_s:
        entry_s = _bold(entry_s)
    if stop_s:
        stop_s = _bold(stop_s)
    if target_s:
        target_s = _bold(target_s)

    print(f"{'entry_price':>18}: {entry_s}")
    print(f"{'stop_price':>18}: {stop_s}")
    print(f"{'target_price':>18}: {target_s}")

    # Shares can come through as int/float/None
    shares_disp = ""
    if shares is not None:
        try:
            shares_disp = str(int(float(shares)))
        except Exception:
            shares_disp = str(shares)

    print(f"{'shares':>18}: {shares_disp}")
    print(f"{'dollars_at_risk':>18}: {_fmt_money(float(risk_dollars)) if risk_dollars is not None else ''}")


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _print_gap_risk_check(result: dict, args: argparse.Namespace) -> None:
    """Premarket gap-risk check (GAP UP + GAP DOWN), with recomputed ticket when --open is provided."""
    if result.get("status") != "planned":
        return

    symbol = str(result.get("symbol") or "")
    planned_entry = _safe_float(result.get("entry_price"))
    planned_stop = _safe_float(result.get("stop_price"))
    if not symbol or planned_entry is None:
        return

    try:
        from project_purple.data_loader import load_symbol_daily
        from project_purple.backtest_v2 import add_atr
        from project_purple.risk import calculate_risk_for_trade
    except Exception as e:
        print("\n=== GAP-RISK CHECK (premarket) ===")
        print("Gap check:  skipped (imports unavailable).")
        print("Note:      ", str(e))
        return

    # Compute ATR
    atr = None
    atr_err = ""
    as_of_input = result.get("as_of_input")
    try:
        df = load_symbol_daily(symbol, as_of_date=as_of_input)
        df = add_atr(df, period=14)
        df = df.dropna(subset=["atr"]).sort_values("date").reset_index(drop=True)
        if not df.empty:
            atr = _safe_float(df.iloc[-1].get("atr"))
    except Exception as e:
        atr = None
        atr_err = str(e)

    open_fill = _safe_float(getattr(args, "open_price", None))

    gap_dir = None  # "UP" | "DOWN" | None
    if open_fill is not None:
        if open_fill > planned_entry:
            gap_dir = "UP"
        elif open_fill < planned_entry:
            gap_dir = "DOWN"

    header = "=== GAP-RISK CHECK (premarket) ==="
    if gap_dir == "UP":
        header = _color(header, "\033[92m")  # green
    elif gap_dir == "DOWN":
        header = _color(header, "\033[91m")  # red
    print("\n" + header)

    if atr is None:
        print("ATR:        (unavailable) — gap check skipped (could not compute ATR from CSV).")
        if atr_err:
            print("Note:       ", atr_err)
        return

    gap_max_pct = float(getattr(args, "gap_max_pct", 0.03))
    gap_max_atr = float(getattr(args, "gap_max_atr", 0.50))

    # GAP UP threshold: choose the lower (stricter) of percent vs ATR units
    pct_limit = planned_entry * (1.0 + gap_max_pct)
    atr_limit = planned_entry + float(atr) * gap_max_atr
    max_entry_gap_up = min(pct_limit, atr_limit)

    # GAP DOWN “too tight to stop” cutoff:
    # If (entry - stop) < 0.33 * ATR => NO TRADE
    too_tight_mult = 0.33
    min_entry_gap_down = None
    if planned_stop is not None:
        min_entry_gap_down = planned_stop + (too_tight_mult * float(atr))

    print(f"Planned entry:        {_fmt_price(planned_entry)}")
    if planned_stop is not None:
        print(f"Planned stop:         {_fmt_price(planned_stop)}")
    print(f"ATR (14d):            {_fmt_price(float(atr))}")

    print(f"Max acceptable entry if GAP UP:   {_fmt_price(max_entry_gap_up)}")
    if min_entry_gap_down is not None:
        # Keep your label (even though mathematically this is a *minimum* acceptable entry)
        print(f"Max acceptable entry if GAP DOWN: {_fmt_price(min_entry_gap_down)}")

    def _print_ticket(title: str, entry_for_ticket: float) -> None:
        rc = result.get("risk_config")
        print(f"\n--- Ticket ({title}) ---")
        if rc is None:
            print("Unavailable: risk_config missing from plan result.")
            return

        try:
            r = calculate_risk_for_trade(
                entry_price=float(entry_for_ticket),
                atr=float(atr),
                risk_config=rc,
                equity=100_000.0,
            )

            print(f"{'entry_price':>18}: {_fmt_price(_safe_float(r.get('entry_price')))}")
            print(f"{'stop_price':>18}: {_fmt_price(_safe_float(r.get('stop_price')))}")
            print(f"{'target_price':>18}: {_fmt_price(_safe_float(r.get('target_price')))}")

            shares = r.get("shares")
            try:
                shares_disp = str(int(float(shares))) if shares is not None else ""
            except Exception:
                shares_disp = str(shares) if shares is not None else ""

            print(f"{'shares':>18}: {shares_disp}")
            print(f"{'dollars_at_risk':>18}: {_fmt_money(_safe_float(r.get('dollar_risk')))}")
        except Exception as e:
            print("Failed to compute:", str(e))

    # If no open/fill price provided: print worst-case boundary tickets for review
    if open_fill is None:
        _print_ticket(f"at max acceptable entry — GAP UP", float(max_entry_gap_up))
        if min_entry_gap_down is not None:
            _print_ticket(f"at max acceptable entry — GAP DOWN", float(min_entry_gap_down))
        print("Open price:           (not provided)")
        print("Action:               At the open, compare your expected fill to the thresholds above.")
        return

    # With open/fill provided: decide SKIP/OK, and if OK print a recomputed ticket using open/fill
    print(f"Open/fill price:      {_fmt_price(open_fill)}")

    # GAP UP fail
    if open_fill > float(max_entry_gap_up):
        print(_color("Action:               SKIP", "\033[91m"), "— gap up too large.")
        _print_ticket("at max acceptable entry — GAP UP (boundary)", float(max_entry_gap_up))
        return

    # GAP DOWN fails (stop proximity)
    if planned_stop is not None:
        if open_fill <= planned_stop:
            print(_color("Action:               SKIP", "\033[91m"), "— entry would be at/below the planned stop.")
            return

        if min_entry_gap_down is not None and open_fill < float(min_entry_gap_down):
            print(_color("Action:               SKIP", "\033[91m"), f"— too tight: (entry - stop) < {too_tight_mult:.2f}*ATR.")
            _print_ticket("at max acceptable entry — GAP DOWN (boundary)", float(min_entry_gap_down))
            return

    # If we got here: within thresholds -> OK, recompute ticket at the actual open/fill
    direction = "GAP UP" if gap_dir == "UP" else ("GAP DOWN" if gap_dir == "DOWN" else "NO GAP")
    _print_ticket(f"recomputed at OPEN/FILL — {direction}", float(open_fill))
    print(_color("Action:               OK", "\033[92m"), "— within thresholds.")


def main() -> int:
    args = _parse_args()
    if args.no_color:
        os.environ["NO_COLOR"] = "1"

    repo_root = _repo_root()
    _print_header(repo_root, args)

    if args.refresh_data or args.refresh_only:
        rc = _run_data_refresh(repo_root)
        if rc != 0:
            return rc
        if args.refresh_only:
            return 0

    # Run trade plan
    try:
        from project_purple.trade_plan import build_trade_plan

        result = build_trade_plan(as_of_date=args.as_of, return_meta=True)
        if result is None:
            result = {"status": "unknown", "note": "build_trade_plan returned None"}
    except Exception as e:
        result = {"status": "launcher_error", "note": str(e)}

    _print_market_state(result)
    _print_trade_plan(result)
    _print_gap_risk_check(result, args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
