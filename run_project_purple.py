# run_project_purple.py
"""
Stable entrypoint for Project Purple.

Why this exists:
- README currently suggests: python -m project_purple.main
- But project_purple/main.py uses local imports like: from risk import RiskConfig
- Those imports work when running as a script from inside the package folder.

This launcher runs main.py in the exact environment it expects
without forcing a multi-file import refactor yet.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    pkg_dir = repo_root / "project_purple"

    if not pkg_dir.exists():
        raise FileNotFoundError(f"Expected folder not found: {pkg_dir}")

    main_path = pkg_dir / "main.py"
    if not main_path.exists():
        raise FileNotFoundError(f"Expected file not found: {main_path}")

    # Mimic "run main.py from inside project_purple/"
    os.chdir(pkg_dir)
    sys.path.insert(0, str(pkg_dir))

    runpy.run_path(str(main_path), run_name="__main__")


if __name__ == "__main__":
    main()
