# run_project_purple.py
"""
Stable entrypoint for Project Purple.

Goal:
- One command can (optionally) refresh data, then run the system end-to-end.

Why:
- main.py currently uses local imports like: from risk import RiskConfig
- Those imports work when running as a script from inside the package folder.

This launcher runs main.py in the exact environment it expects, and can also
invoke the downloader first so the run uses the freshest CSVs.
"""

from __future__ import annotations

import argparse
import os
import runpy
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Project Purple end-to-end.")
    p.add_argument(
        "--refresh-data",
        action="store_true",
        help="Refresh CSVs via data_downloader.py before running main.py",
    )
    p.add_argument(
        "--symbols",
        nargs="*",
        default=None,
        help="Optional symbols to refresh (e.g., --symbols SPY AAPL). "
             "If omitted, downloader decides default (usually CANDIDATE_SYMBOLS).",
    )
    p.add_argument("--period", default="10y", help="yfinance period (default: 10y)")
    p.add_argument("--interval", default="1d", help="yfinance interval (default: 1d)")
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Pass --overwrite to the downloader (only if your downloader supports it).",
    )
    p.add_argument(
        "--refresh-only",
        action="store_true",
        help="Only refresh data; do not run main.py.",
    )
    return p.parse_args()


def _run_downloader(repo_root: Path, pkg_dir: Path, args: argparse.Namespace) -> None:
    """
    Run the downloader as a subprocess so we always execute the exact file
    you have in PyCharm (no import/path ambiguity).
    """
    downloader_path = pkg_dir / "data_downloader.py"
    if not downloader_path.exists():
        raise FileNotFoundError(f"Expected file not found: {downloader_path}")

    cmd: List[str] = [sys.executable, "-u", str(downloader_path)]
    if args.symbols:
        cmd += ["--symbols", *args.symbols]

    # These flags exist in the version you pasted; if your local file differs,
    # the downloader will print an argparse error (and we should fix it next).
    if args.period:
        cmd += ["--period", args.period]
    if args.interval:
        cmd += ["--interval", args.interval]
    if args.overwrite:
        cmd += ["--overwrite"]

    print("\n=== REFRESH DATA ===")
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(repo_root), check=True)


def _run_main(repo_root: Path, pkg_dir: Path) -> None:
    main_path = pkg_dir / "main.py"
    if not main_path.exists():
        raise FileNotFoundError(f"Expected file not found: {main_path}")

    # Mimic "run main.py from inside project_purple/"
    os.chdir(pkg_dir)
    sys.path.insert(0, str(pkg_dir))

    print("\n=== RUN MAIN ===")
    runpy.run_path(str(main_path), run_name="__main__")


def main() -> None:
    args = _parse_args()

    repo_root = Path(__file__).resolve().parent
    pkg_dir = repo_root / "project_purple"

    if not pkg_dir.exists():
        raise FileNotFoundError(f"Expected folder not found: {pkg_dir}")

    if args.refresh_data:
        _run_downloader(repo_root=repo_root, pkg_dir=pkg_dir, args=args)

    if args.refresh_only:
        print("\nRefresh-only requested. Exiting without running main.")
        return

    _run_main(repo_root=repo_root, pkg_dir=pkg_dir)


if __name__ == "__main__":
    main()
