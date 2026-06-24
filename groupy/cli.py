"""Command-line interface for Groupy.

The legacy interactive menu is still available, but new subcommands should call
the library API directly so they can also be reused by future GUI code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


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
        help="count molecular groups",
    )
    count_input = count_parser.add_mutually_exclusive_group(required=True)
    count_input.add_argument(
        "--smiles",
        help="SMILES string to analyze",
    )
    count_input.add_argument(
        "--input",
        type=Path,
        help="txt, csv, or xlsx file containing SMILES values",
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

    calculate_parser = subparsers.add_parser(
        "calculate",
        help="calculate molecular properties",
    )
    calculate_input = calculate_parser.add_mutually_exclusive_group(required=True)
    calculate_input.add_argument(
        "--smiles",
        help="SMILES string to analyze",
    )
    calculate_input.add_argument(
        "--input",
        type=Path,
        help="txt, csv, or xlsx file containing SMILES values",
    )
    calculate_parser.add_argument(
        "--parameter-type",
        choices=["step_wise", "simultaneous"],
        default="step_wise",
        help="group contribution parameter type",
    )
    calculate_parser.add_argument(
        "--no-check-hydrocarbon",
        action="store_true",
        help="calculate combustion-related properties without hydrocarbon filtering",
    )
    calculate_parser.add_argument(
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
    from groupy.api import count_many_smiles, count_smiles, load_smiles_file, write_records_csv

    if args.smiles:
        records = [
            count_smiles(
                args.smiles,
                include_zero=args.include_zero,
                include_smiles=not args.no_smiles,
            )
        ]
    else:
        records = count_many_smiles(
            load_smiles_file(args.input),
            include_zero=args.include_zero,
            include_smiles=not args.no_smiles,
        )

    if args.output:
        write_records_csv(records, args.output)
    else:
        print(_format_json_output(records, single=bool(args.smiles)))

    return 0


def _run_calculate(args: argparse.Namespace) -> int:
    from groupy.api import calculate_many_smiles, calculate_smiles, load_smiles_file, write_records_csv

    check_hydrocarbon = not args.no_check_hydrocarbon
    if args.smiles:
        records = [
            calculate_smiles(
                args.smiles,
                check_hydrocarbon=check_hydrocarbon,
                parameter_type=args.parameter_type,
            )
        ]
    else:
        records = calculate_many_smiles(
            load_smiles_file(args.input),
            check_hydrocarbon=check_hydrocarbon,
            parameter_type=args.parameter_type,
        )

    if args.output:
        write_records_csv(records, args.output)
    else:
        print(_format_json_output(records, single=bool(args.smiles)))

    return 0


def _format_json_output(records: list[dict], *, single: bool) -> str:
    payload = records[0] if single else records
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None or args.command == "interactive":
        return _run_legacy_interactive()
    if args.command == "count":
        return _run_count(args)
    if args.command == "calculate":
        return _run_calculate(args)

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
