"""Programmatic API helpers for non-interactive Groupy workflows."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from groupy.gp_calculator import Calculator
from groupy.gp_counter import Counter


def load_smiles_file(path: str | Path) -> list[str]:
    """Load SMILES strings from a txt, csv, or xlsx file."""
    input_path = Path(path)
    suffix = input_path.suffix.lower()

    if suffix == ".txt":
        with input_path.open(encoding="utf-8") as file:
            smiles = [line.strip() for line in file]
    elif suffix == ".csv":
        smiles = _load_smiles_column(pd.read_csv(input_path), input_path)
    elif suffix == ".xlsx":
        smiles = _load_smiles_column(pd.read_excel(input_path), input_path)
    else:
        raise ValueError("Input file must use .txt, .csv, or .xlsx format.")

    return [item for item in smiles if item]


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


def write_records_csv(records: list[dict[str, Any]], path: str | Path) -> None:
    """Write result records to a CSV file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output_path, index=False)


def _load_smiles_column(dataframe: pd.DataFrame, path: Path) -> list[str]:
    if "smiles" not in dataframe.columns:
        raise ValueError(f"{path} must contain a 'smiles' column.")
    return [str(value).strip() for value in dataframe["smiles"].dropna()]
