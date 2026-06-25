"""Build a double-clickable Windows desktop app for Groupy with PyInstaller."""

from __future__ import annotations

import argparse
import importlib.util
import platform
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENTRY_SCRIPT = ROOT / "scripts" / "groupy_gui_entry.py"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="Groupy", help="Application name. Default: Groupy.")
    parser.add_argument(
        "--mode",
        choices=("onedir", "onefile"),
        default="onedir",
        help="Build an app folder or a single executable. Default: onedir.",
    )
    parser.add_argument("--dist-path", type=Path, default=ROOT / "dist", help="PyInstaller dist directory.")
    parser.add_argument(
        "--work-path",
        type=Path,
        default=ROOT / "build" / "pyinstaller",
        help="PyInstaller temporary work directory.",
    )
    parser.add_argument(
        "--spec-path",
        type=Path,
        default=ROOT / "build" / "pyinstaller",
        help="Directory for generated PyInstaller spec files.",
    )
    parser.add_argument("--console", action="store_true", help="Keep a console window for debugging.")
    parser.add_argument("--no-clean", action="store_true", help="Do not clear PyInstaller cache before building.")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without running PyInstaller.")
    args = parser.parse_args(argv)

    command = build_command(args)
    expected_output = expected_output_path(args)

    if args.dry_run:
        print("PyInstaller command:")
        print(format_command(command))
        print(f"Expected output: {expected_output}")
        return 0

    missing = [
        module
        for module in ("PyInstaller", "PySide6")
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        print(
            "Missing build dependencies: {}. Install them with "
            '`python -m pip install -e ".[gui,package]"`.'.format(", ".join(missing)),
            file=sys.stderr,
        )
        return 1

    if platform.system() != "Windows":
        print(
            "Warning: PyInstaller builds for the current platform only. "
            "Run this script on Windows to produce a Windows .exe.",
            file=sys.stderr,
        )

    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode != 0:
        return completed.returncode

    print(f"Built Groupy desktop app: {expected_output}")
    return 0


def build_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        f"--name={args.name}",
        f"--distpath={args.dist_path}",
        f"--workpath={args.work_path}",
        f"--specpath={args.spec_path}",
        "--collect-data=groupy",
        "--collect-data=rdkit",
        "--collect-submodules=rdkit",
        f"--{args.mode}",
    ]

    if args.console:
        command.append("--console")
    else:
        command.append("--windowed")

    if not args.no_clean:
        command.append("--clean")

    command.append(str(ENTRY_SCRIPT))
    return command


def expected_output_path(args: argparse.Namespace) -> Path:
    if args.mode == "onefile":
        return args.dist_path / f"{args.name}.exe"
    return args.dist_path / args.name / f"{args.name}.exe"


def format_command(command: list[str]) -> str:
    return shlex.join(str(part) for part in command)


if __name__ == "__main__":
    raise SystemExit(main())
