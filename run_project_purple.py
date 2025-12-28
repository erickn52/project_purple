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

    # Gap-risk guard
    p.add_argument(
        "--open",
        dest="open_price",
        type=float,
        default=None,
        help=(
            "Optional actual OPEN/FILL price you expect to enter at. "
            "If provided, Project Purple will evaluate gap-up OR gap-down risk and print a recomputed ticket."
        ),
    )

    # ONE-BUTTON default: open a small window to type the open/fill price without re-running
    p.add_argument(
        "--open-window",
        dest="open_window",
        action="store_true",
        default=True,
        help="Show a small window after the plan prints so you can type OPEN/FILL price and compute the ticket (default: ON).",
    )
    p.add_argument(
        "--no-open-window",
        dest="open_window",
        action="store_false",
        help="Disable the open/fill window and only print thresholds/tickets.",
    )

    p.add_argument(
        "--equity",
        type=float,
        default=100_000.0,
        help="Equity used for position sizing in recomputed tickets. Default: 100000.0",
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

    regime_label = str(regime).upper()
    if regime_label == "BULL":
        regime_label = _color(regime_label, "\033[92m")
    elif regime_label == "BEAR":
        regime_label = _color(regime_label, "\033[91m")
    elif "CHOP" in regime_label:
        regime_label = _color(regime_label, "\033[93m")
    else:
        regime_label = _color(regime_label, "\033[90m")

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

    if entry_s:
        entry_s = _bold(entry_s)
    if stop_s:
        stop_s = _bold(stop_s)
    if target_s:
        target_s = _bold(target_s)

    print(f"{'entry_price':>18}: {entry_s}")
    print(f"{'stop_price':>18}: {stop_s}")
    print(f"{'target_price':>18}: {target_s}")

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


def _compute_ticket_lines(calculate_risk_for_trade, entry_for_ticket: float, atr: float, risk_config: Any, equity: float) -> list[str]:
    r = calculate_risk_for_trade(
        entry_price=float(entry_for_ticket),
        atr=float(atr),
        risk_config=risk_config,
        equity=float(equity),
    )

    entry_p = _safe_float(r.get("entry_price"))
    stop_p = _safe_float(r.get("stop_price"))
    target_p = _safe_float(r.get("target_price"))
    shares = r.get("shares")
    dollar_risk = _safe_float(r.get("dollar_risk"))  # matches your existing launcher usage

    try:
        shares_disp = str(int(float(shares))) if shares is not None else ""
    except Exception:
        shares_disp = str(shares) if shares is not None else ""

    return [
        f"{'entry_price':>18}: {_fmt_price(entry_p)}",
        f"{'stop_price':>18}: {_fmt_price(stop_p)}",
        f"{'target_price':>18}: {_fmt_price(target_p)}",
        f"{'shares':>18}: {shares_disp}",
        f"{'dollars_at_risk':>18}: {_fmt_money(dollar_risk)}",
    ]


def _open_price_window(
    *,
    symbol: str,
    planned_entry: float,
    planned_stop: Optional[float],
    atr: float,
    max_entry_gap_up: float,
    min_entry_gap_down: Optional[float],
    too_tight_mult: float,
    result: dict,
    args: argparse.Namespace,
    calculate_risk_for_trade,
) -> None:
    """
    Small window to type OPEN/FILL price and compute ticket without rerunning the plan.
    Blocks until closed.
    """
    try:
        import tkinter as tk
        from tkinter import messagebox
    except Exception as e:
        print("\n=== OPEN/FILL WINDOW ===")
        print("Tkinter not available; cannot open window.")
        print("Note:", str(e))
        print("Tip: run with --open 123.45")
        return

    risk_config = result.get("risk_config")
    if risk_config is None:
        print("\n=== OPEN/FILL WINDOW ===")
        print("Cannot open window: risk_config missing from plan result.")
        return

    equity = float(getattr(args, "equity", 100_000.0))

    root = tk.Tk()
    root.title("Project Purple — Open/FILL Ticket")
    root.geometry("720x420")
    root.minsize(720, 420)

    # Top info
    header = tk.Label(root, text=f"{symbol} — Planned entry {planned_entry:.2f}", font=("Segoe UI", 12, "bold"))
    header.pack(pady=(10, 4))

    info_lines = [
        f"Planned stop: {_fmt_price(planned_stop)}" if planned_stop is not None else "Planned stop: (none)",
        f"ATR (14d): {_fmt_price(atr)}",
        f"Max acceptable entry if GAP UP: {_fmt_price(max_entry_gap_up)}",
        f"Min acceptable entry if GAP DOWN: {_fmt_price(min_entry_gap_down) if min_entry_gap_down is not None else '(n/a)'}",
        f"Equity for sizing: {_fmt_money(equity)}",
    ]
    info = tk.Label(root, text="   |   ".join(info_lines), font=("Segoe UI", 9))
    info.pack(pady=(0, 10))

    # Input row
    row = tk.Frame(root)
    row.pack(pady=(0, 10))

    tk.Label(row, text="OPEN/FILL price:", font=("Segoe UI", 10)).pack(side=tk.LEFT, padx=(0, 8))
    price_var = tk.StringVar(value="")
    entry_box = tk.Entry(row, textvariable=price_var, width=14, font=("Segoe UI", 10))
    entry_box.pack(side=tk.LEFT)
    entry_box.focus_set()

    # Output area
    out = tk.Text(root, height=14, width=90, font=("Consolas", 10))
    out.pack(padx=10, pady=(5, 10), fill=tk.BOTH, expand=True)

    def write_output(text: str) -> None:
        out.delete("1.0", tk.END)
        out.insert(tk.END, text)

    def compute() -> None:
        open_fill = _safe_float(price_var.get().strip())
        if open_fill is None:
            messagebox.showerror("Invalid price", "Please enter a valid number like 187.42")
            return

        gap_pct = (open_fill - planned_entry) / planned_entry if planned_entry != 0 else None
        gap_dir = "UP" if open_fill > planned_entry else ("DOWN" if open_fill < planned_entry else "FLAT")
        direction = f"{gap_dir} ({_fmt_pct_decimal(gap_pct) if gap_pct is not None else ''})"

        lines: list[str] = []
        lines.append("=== GAP-RISK DECISION ===")
        lines.append(f"Symbol:              {symbol}")
        lines.append(f"Open/FILL:           {_fmt_price(open_fill)}")
        lines.append(f"Planned entry:       {_fmt_price(planned_entry)}")
        if planned_stop is not None:
            lines.append(f"Planned stop:        {_fmt_price(planned_stop)}")
        lines.append(f"Direction:           {direction}")
        lines.append("")

        # GAP UP fail
        if open_fill > max_entry_gap_up:
            lines.append("Action:              SKIP — gap up too large.")
            lines.append("")
            lines.append(f"--- Ticket (boundary at max GAP UP {_fmt_price(max_entry_gap_up)}) ---")
            try:
                lines.extend(_compute_ticket_lines(calculate_risk_for_trade, max_entry_gap_up, atr, risk_config, equity))
            except Exception as e:
                lines.append(f"Failed to compute boundary ticket: {e}")
            text = "\n".join(lines)
            write_output(text)
            print("\n" + text)
            return

        # GAP DOWN fails
        if planned_stop is not None:
            if open_fill <= planned_stop:
                lines.append("Action:              SKIP — entry would be at/below the planned stop.")
                text = "\n".join(lines)
                write_output(text)
                print("\n" + text)
                return

            if min_entry_gap_down is not None and open_fill < min_entry_gap_down:
                lines.append(f"Action:              SKIP — too tight: (entry - stop) < {too_tight_mult:.2f}*ATR.")
                lines.append("")
                lines.append(f"--- Ticket (boundary at min GAP DOWN {_fmt_price(min_entry_gap_down)}) ---")
                try:
                    lines.extend(_compute_ticket_lines(calculate_risk_for_trade, min_entry_gap_down, atr, risk_config, equity))
                except Exception as e:
                    lines.append(f"Failed to compute boundary ticket: {e}")
                text = "\n".join(lines)
                write_output(text)
                print("\n" + text)
                return

        # OK: recompute at actual open/fill
        lines.append("Action:              OK — within thresholds.")
        lines.append("")
        lines.append(f"--- Ticket (recomputed at OPEN/FILL {_fmt_price(open_fill)}) ---")
        try:
            lines.extend(_compute_ticket_lines(calculate_risk_for_trade, open_fill, atr, risk_config, equity))
        except Exception as e:
            lines.append(f"Failed to compute ticket: {e}")

        text = "\n".join(lines)
        write_output(text)
        print("\n" + text)

    def clear() -> None:
        price_var.set("")
        write_output("")
        entry_box.focus_set()

    btns = tk.Frame(root)
    btns.pack(pady=(0, 10))
    tk.Button(btns, text="Compute Ticket", command=compute, width=16).pack(side=tk.LEFT, padx=6)
    tk.Button(btns, text="Clear", command=clear, width=10).pack(side=tk.LEFT, padx=6)
    tk.Button(btns, text="Close", command=root.destroy, width=10).pack(side=tk.LEFT, padx=6)

    root.bind("<Return>", lambda _e: compute())

    # Helpful initial message
    write_output(
        "Type/paste an OPEN/FILL price above, then click 'Compute Ticket' (or press Enter).\n"
        "You can recompute as many times as you want without rerunning Project Purple.\n"
    )

    root.mainloop()


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

    print("\n=== GAP-RISK CHECK (premarket) ===")

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
        print(f"Min acceptable entry if GAP DOWN: {_fmt_price(min_entry_gap_down)}")

    open_fill = _safe_float(getattr(args, "open_price", None))

    # If user didn't provide --open: open the window (default behavior) so they can type price without rerunning
    if open_fill is None and bool(getattr(args, "open_window", True)):
        _open_price_window(
            symbol=symbol,
            planned_entry=float(planned_entry),
            planned_stop=planned_stop,
            atr=float(atr),
            max_entry_gap_up=float(max_entry_gap_up),
            min_entry_gap_down=float(min_entry_gap_down) if min_entry_gap_down is not None else None,
            too_tight_mult=float(too_tight_mult),
            result=result,
            args=args,
            calculate_risk_for_trade=calculate_risk_for_trade,
        )
        return

    # If --open provided, we still compute in-console (useful for automation)
    if open_fill is None:
        print("Open/fill price:      (not provided)")
        print("Action:               Provide --open 123.45 (or keep default window enabled).")
        return

    # If we have open_fill, just compute decision quickly to console (non-window path)
    gap_dir = "UP" if open_fill > planned_entry else ("DOWN" if open_fill < planned_entry else "FLAT")
    gap_pct = (open_fill - planned_entry) / planned_entry if planned_entry != 0 else None

    print("\n=== GAP-RISK DECISION ===")
    print(f"Open/fill price:      {_fmt_price(open_fill)}")
    if gap_pct is not None:
        print(f"Gap vs planned entry: {_fmt_pct_decimal(gap_pct)}")
    print(f"Direction:            {gap_dir}")

    if open_fill > float(max_entry_gap_up):
        print(_color("Action:               SKIP", "\033[91m"), "— gap up too large.")
        return

    if planned_stop is not None:
        if open_fill <= planned_stop:
            print(_color("Action:               SKIP", "\033[91m"), "— entry would be at/below the planned stop.")
            return
        if min_entry_gap_down is not None and open_fill < float(min_entry_gap_down):
            print(_color("Action:               SKIP", "\033[91m"), f"— too tight: (entry - stop) < {too_tight_mult:.2f}*ATR.")
            return

    print(_color("Action:               OK", "\033[92m"), "— within thresholds.")
    # Ticket printing could be added here too, but window path is the default for your workflow.


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
