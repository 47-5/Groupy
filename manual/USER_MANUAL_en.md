# Groupy User Manual

Last updated: 2026-06-25

This manual describes the current refactored Groupy workflow, including the desktop GUI, command-line interface, Python API, input formats, optional advanced features, and Windows packaging. The older background-oriented manual remains available at `manual/Groupy_manual.md`.

## 1. What Groupy Does

Groupy is a SMILES-based molecular analysis toolkit. It can:

- Calculate molecular properties from SMILES.
- Count group-contribution groups from SMILES.
- Export tabular results to CSV.
- Show a 2D structure preview in the desktop GUI.
- Import SMILES from `.txt`, `.csv`, and `.xlsx` files.
- Be used as a Python package in scripts or other applications.
- Optionally use OpenBabel for file conversion and Gaussian input generation.

Ordinary users should start with the desktop GUI. Script users and developers can use the CLI or Python API.

## 2. Recommended Usage

### 2.1 Ordinary Users

If you received a packaged Windows app, open:

```text
dist/Groupy/Groupy.exe
```

Double-click `Groupy.exe` to start the app. Do not copy only `Groupy.exe`; keep the entire `dist/Groupy` folder because `_internal` contains the Python runtime, RDKit, Qt, OpenBLAS, and other required libraries.

### 2.2 Developers And Script Users

If you run from source, create an isolated environment:

```powershell
conda create -n groupy_env -c conda-forge python=3.11 rdkit pandas numpy openpyxl tqdm joblib
conda activate groupy_env
python -m pip install -e .
```

For GUI support:

```powershell
python -m pip install -e ".[gui]"
```

For Windows app packaging:

```powershell
python -m pip install -e ".[gui,package]"
```

For OpenBabel-dependent features:

```powershell
conda install -c conda-forge openbabel
```

Avoid `pip install openbabel`, especially on Windows.

## 3. Desktop GUI

Start the GUI with:

```powershell
Groupy-GUI
```

Or double-click the packaged `dist/Groupy/Groupy.exe`.

### 3.1 Enter SMILES

Enter one or more SMILES strings in the SMILES editor, one per line:

```text
C1CCCC1
CCO
CC(C)C
```

The first valid SMILES in the editor is shown in the 2D structure preview.

### 3.2 Import SMILES From A File

Click `Import File` and choose a SMILES input file. Supported formats:

- `.txt`
- `.csv`
- `.xlsx`

A `.txt` file should contain one SMILES per line:

```text
C1CCCC1
CCO
CC(C)C
```

`.csv` and `.xlsx` files must contain a column named `smiles`:

```text
smiles
C1CCCC1
CCO
CC(C)C
```

After import, the SMILES values are inserted into the editor.

### 3.3 2D Structure Preview

The right-hand preview panel shows the 2D structure for a SMILES string.

- While editing input, it previews the first valid SMILES.
- After calculation or group counting, selecting a table row updates the preview to that molecule.
- If a SMILES string cannot be parsed, the preview shows `Invalid SMILES`.

This feature uses RDKit only. It does not require OpenBabel or ASE.

### 3.4 Calculate Molecular Properties

Click `Calculate Properties` to calculate properties. Results appear in the table and can be exported to CSV.

Calculation options:

- `Parameters`
  - `Stepwise`: recommended default.
  - `Simultaneous`: available for compatibility, but may be unreliable for some molecules.
- `Hydrocarbon check`
  - Enabled: combustion-related properties are calculated only for hydrocarbons.
  - Disabled: Groupy attempts to calculate those properties for non-hydrocarbons too, but the results may not be physically meaningful.

### 3.5 Count Groups

Click `Count Groups` to count group-contribution groups.

Counting options:

- `Show zero groups`
  - Enabled: include all groups, including zero-count groups.
  - Disabled: include only nonzero groups.
- `Include SMILES`
  - Enabled: include a SMILES column in the output.

### 3.6 Export CSV

After calculation or group counting, click `Export CSV` to save the current table.

The exported CSV can be opened with Excel, WPS, LibreOffice, or Python/pandas.

### 3.7 GUI Troubleshooting

If running from source and PySide6 is missing:

```powershell
python -m pip install -e ".[gui]"
```

If the packaged executable does not start:

- Make sure the whole `dist/Groupy` folder is present.
- Make sure `_internal` is next to `Groupy.exe`.
- Try copying `dist/Groupy` to a simple path without unusual characters.
- Run `dist\Groupy\Groupy.exe` from a terminal to see error messages.

## 4. Command-Line Interface

After installation, the `Groupy` command is available.

### 4.1 Help

```powershell
Groupy --help
Groupy count --help
Groupy calculate --help
Groupy convert --help
```

### 4.2 Count Groups For One SMILES

```powershell
Groupy count --smiles C1CCCC1
```

Example JSON output:

```json
{"f_168": 5, "smiles": "C1CCCC1"}
```

Write CSV output:

```powershell
Groupy count --smiles C1CCCC1 --output count.csv
```

Include zero-count groups:

```powershell
Groupy count --smiles C1CCCC1 --include-zero --output count_full.csv
```

### 4.3 Calculate Properties For One SMILES

```powershell
Groupy calculate --smiles C1CCCC1
```

Write CSV output:

```powershell
Groupy calculate --smiles C1CCCC1 --output calculate.csv
```

Select parameter type:

```powershell
Groupy calculate --smiles C1CCCC1 --parameter-type step_wise
Groupy calculate --smiles C1CCCC1 --parameter-type simultaneous
```

Disable hydrocarbon filtering:

```powershell
Groupy calculate --smiles CCO --no-check-hydrocarbon
```

### 4.4 Batch Calculation

Input files can be `.txt`, `.csv`, or `.xlsx`:

```powershell
Groupy calculate --input SMILES.txt --output calculate.csv
Groupy count --input SMILES.txt --output count.csv
```

`.csv` and `.xlsx` files must contain a `smiles` column.

### 4.5 File Conversion

Single-file conversion:

```powershell
Groupy convert --input molecule.xyz --from xyz --to mol2 --output molecule.mol2
```

This requires OpenBabel:

```powershell
conda install -c conda-forge openbabel
```

Conversion is an optional advanced workflow and is not part of the default GUI workflow.

### 4.6 Legacy Interactive Menu

Running:

```powershell
Groupy
```

starts the legacy interactive menu. New scripts and automation should prefer:

- `Groupy count`
- `Groupy calculate`
- `Groupy convert`
- `groupy.api`

## 5. Python API

Groupy can be used as a Python library.

### 5.1 Single-Molecule Calls

```python
from groupy.api import calculate_smiles, count_smiles

properties = calculate_smiles("C1CCCC1")
groups = count_smiles("C1CCCC1")

print(properties)
print(groups)
```

### 5.2 Batch Calls

```python
from groupy.api import calculate_many_smiles, count_many_smiles, write_records_csv

smiles_values = ["C1CCCC1", "CCO", "CC(C)C"]

properties = calculate_many_smiles(smiles_values)
groups = count_many_smiles(smiles_values)

write_records_csv(properties, "calculate.csv")
write_records_csv(groups, "count.csv")
```

### 5.3 Load SMILES From A File

```python
from groupy.io import load_smiles_file

smiles_values = load_smiles_file("SMILES.xlsx")
```

`.csv` and `.xlsx` files must contain a `smiles` column.

### 5.4 Lower-Level Classes

For compatibility or finer control, use the lower-level classes:

```python
from groupy.gp_calculator import Calculator
from groupy.gp_counter import Counter

calculator = Calculator()
counter = Counter()

calculator.calculate_mols(
    "SMILES.txt",
    properties_file_path="calculate.csv",
    verbose=False,
)

counter.count_mols(
    "SMILES.txt",
    count_result_file_path="count.csv",
    add_smiles=True,
    verbose=False,
)
```

Batch APIs continue after invalid SMILES by default for backward compatibility. To stop on the first error:

```python
calculator.calculate_mols(
    "SMILES.txt",
    properties_file_path="calculate.csv",
    continue_on_error=False,
    verbose=False,
)
```

## 6. Optional Advanced Features

### 6.1 OpenBabel Conversion

OpenBabel-dependent features include:

- Generating `.xyz` files from SMILES.
- Converting between common molecular file formats.
- Extracting SMILES from structure files.
- Supporting coordinate generation for Gaussian input files.

Install OpenBabel with:

```powershell
conda install -c conda-forge openbabel
```

### 6.2 Gaussian Input Generation

`groupy.gp_generator.Generator` can generate Gaussian `.gjf` files from SMILES. This workflow usually depends on OpenBabel for 3D coordinate generation.

This is currently an advanced scripting workflow, not part of the default GUI workflow.

### 6.3 Molecular Visualization

The legacy Viewer depends on ASE:

```powershell
python -m pip install -e ".[viewer]"
```

Or:

```powershell
conda install -c conda-forge ase
```

The current GUI 2D structure preview does not require ASE.

## 7. Build A Windows App

To distribute a double-clickable Windows app, use a clean conda-forge OpenBLAS packaging environment:

```powershell
conda create -n groupy_package -c conda-forge python=3.11 rdkit pandas numpy openpyxl tqdm joblib pyside6 pyinstaller "libblas=*=*openblas"
conda activate groupy_package
python -m pip install -e . --no-deps
python scripts\build_windows_app.py
```

Default output:

```text
dist/Groupy/Groupy.exe
```

Distribute the entire `dist/Groupy` folder, not just the executable.

The build script removes the previous `dist/Groupy` output by default so stale `_internal` files do not affect package size. Use this only for PyInstaller debugging:

```powershell
python scripts\build_windows_app.py --no-clean-dist
```

Package size notes:

```text
PACKAGING_SIZE_REPORT.md
```

Release checklist:

```text
RELEASE_CHECKLIST.md
```

## 8. Known Limitations

- The `simultaneous` parameter type may be unreliable for some molecules. The recommended default is `step_wise`.
- Combustion enthalpy, heat value, and specific impulse are designed mainly for hydrocarbons. Interpret non-hydrocarbon results carefully.
- OpenBabel conversion, Gaussian input generation, and ASE visualization are optional advanced workflows and are not part of the default GUI workflow.
- The packaged `_internal` folder contains runtime dependencies. Do not delete DLLs manually unless the executable is fully retested.
- More advanced GUI pages, such as OpenBabel conversion and Gaussian input generation, can be added in a later phase.

## 9. Troubleshooting

### 9.1 SMILES Cannot Be Parsed

Check:

- Extra spaces or invisible characters.
- Ring closure digits.
- Atom symbol capitalization.
- Correct lowercase aromatic atoms.

### 9.2 CSV Or XLSX Import Fails

Make sure the file contains a lowercase `smiles` column:

```text
smiles
```

### 9.3 Packaged App Is Too Large

Use the OpenBLAS packaging environment:

```powershell
conda create -n groupy_package -c conda-forge python=3.11 rdkit pandas numpy openpyxl tqdm joblib pyside6 pyinstaller "libblas=*=*openblas"
```

If `_internal` contains many `mkl_*.dll` files, the build environment is still linked against Intel MKL.

### 9.4 GUI Does Not Start

When running from source:

```powershell
Groupy-GUI --check
```

If PySide6 is missing:

```powershell
python -m pip install -e ".[gui]"
```

For packaged apps, first check that the entire `dist/Groupy` folder is complete.
