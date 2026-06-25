"""Programmatic API helpers for non-interactive Groupy workflows."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from groupy.exceptions import ConversionError
from groupy.gp_calculator import Calculator
from groupy.gp_counter import Counter
from groupy.io import load_smiles_file, write_records_csv


def count_smiles(
    smiles: str,
    *,
    include_zero: bool = False,
    include_smiles: bool = True,
    counter: Counter | None = None,
) -> dict[str, Any]:
    """Count molecular groups for one SMILES string."""
    active_counter = counter or Counter()
    return active_counter.count_a_mol(
        smiles,
        clear_mode=not include_zero,
        add_smiles=include_smiles,
    )


def count_many_smiles(
    smiles_values: Iterable[str],
    *,
    include_zero: bool = False,
    include_smiles: bool = True,
) -> list[dict[str, Any]]:
    """Count molecular groups for many SMILES strings."""
    counter = Counter()
    return [
        count_smiles(
            smiles,
            include_zero=include_zero,
            include_smiles=include_smiles,
            counter=counter,
        )
        for smiles in smiles_values
    ]


def calculate_smiles(
    smiles: str,
    *,
    check_hydrocarbon: bool = True,
    parameter_type: str = "step_wise",
    calculator: Calculator | None = None,
) -> dict[str, Any]:
    """Calculate properties for one SMILES string."""
    active_calculator = calculator or Calculator()
    return active_calculator.calculate_a_mol(
        smiles,
        check_hydrocarbon=check_hydrocarbon,
        parameter_type=parameter_type,
    )


def calculate_many_smiles(
    smiles_values: Iterable[str],
    *,
    check_hydrocarbon: bool = True,
    parameter_type: str = "step_wise",
) -> list[dict[str, Any]]:
    """Calculate properties for many SMILES strings."""
    calculator = Calculator()
    return [
        calculate_smiles(
            smiles,
            check_hydrocarbon=check_hydrocarbon,
            parameter_type=parameter_type,
            calculator=calculator,
        )
        for smiles in smiles_values
    ]


def convert_file(
    input_path: str | Path,
    from_format: str,
    to_format: str,
    output_path: str | Path | None = None,
) -> Path:
    """Convert one molecular file and return the output path."""
    from groupy.gp_convertor import Convertor

    input_file = Path(input_path)
    target_format = to_format.lstrip(".")
    output_file = Path(output_path) if output_path is not None else input_file.with_suffix(f".{target_format}")
    converted_path = Convertor.convert_file_type(
        in_format=from_format.lstrip("."),
        in_path=str(input_file),
        out_format=target_format,
        out_path=str(output_file),
    )
    if converted_path is None or not output_file.exists():
        raise ConversionError(f"Failed to convert {input_file} to {output_file}.")
    return Path(converted_path)
