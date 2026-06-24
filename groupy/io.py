"""Shared file I/O helpers for Groupy."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


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


def write_records_csv(records: list[dict[str, Any]], path: str | Path) -> None:
    """Write result records to a CSV file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output_path, index=False)


def write_text_lines(lines: list[str], path: str | Path) -> None:
    """Write one text item per line."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for line in lines:
            file.write(f"{line}\n")


def _load_smiles_column(dataframe: pd.DataFrame, path: Path) -> list[str]:
    if "smiles" not in dataframe.columns:
        raise ValueError(f"{path} must contain a 'smiles' column.")
    return [str(value).strip() for value in dataframe["smiles"].dropna()]
