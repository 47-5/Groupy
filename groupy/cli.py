"""Command-line interface for Groupy.

The legacy interactive menu is still available, but new subcommands should call
the library API directly so they can also be reused by future GUI code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="Groupy",
        description="Groupy command-line tools for molecular analysis.",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser(
        "interactive",
        help="start the legacy interactive menu",
    )

    count_parser = subparsers.add_parser(
        "count",
        help="count molecular groups for a SMILES string",
    )
    count_parser.add_argument(
        "--smiles",
        required=True,
        help="SMILES string to analyze",
    )
    count_parser.add_argument(
        "--include-zero",
        action="store_true",
        help="include groups with zero counts in the output",
    )
    count_parser.add_argument(
        "--no-smiles",
        action="store_true",
        help="omit the input SMILES from the output",
    )
    count_parser.add_argument(
        "--output",
        type=Path,
        help="optional CSV path for saving the result",
    )

    return parser


def _run_legacy_interactive() -> int:
    from groupy.groupy_main import main as legacy_main

    legacy_main()
    return 0


def _run_count(args: argparse.Namespace) -> int:
    from groupy.gp_counter import Counter

    result = Counter().count_a_mol(
        args.smiles,
        clear_mode=not args.include_zero,
        add_smiles=not args.no_smiles,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([result]).to_csv(args.output, index=False)
    else:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None or args.command == "interactive":
        return _run_legacy_interactive()
    if args.command == "count":
        return _run_count(args)

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
