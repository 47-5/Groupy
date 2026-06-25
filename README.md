This repository corresponds to the paper *Groupy: An open-source toolkit for molecular simulation and property calculation*

### Install

Download the source code:

`git clone https://github.com/47-5/Groupy.git`

One may create an environment using Anaconda:

`conda create -n groupy_env python=3.10`

`conda activate groupy_env`

Install for development:

`python -m pip install -e .`

Install visualization support when needed:

`python -m pip install -e ".[viewer]"`

Install desktop GUI support when needed:

`python -m pip install -e ".[gui]"`

Install Windows app packaging support when needed:

`python -m pip install -e ".[gui,package]"`

Install conversion and Gaussian input generation support when needed:

`conda install -c conda-forge openbabel` (**Do not** use `pip install openbabel`.)

Then one can enter `Groupy` in terminal to start Groupy.
OpenBabel is only required for conversion and Gaussian input generation workflows.
ASE is only required for molecular visualization workflows.

For non-interactive use, one can run commands such as:

`Groupy count --smiles C1CCCC1`

`Groupy calculate --smiles C1CCCC1`

`Groupy convert --input molecule.xyz --from xyz --to mol2 --output molecule.mol2`

To launch the desktop GUI after installing GUI support:

`Groupy-GUI`

The GUI supports SMILES text input, SMILES file import, 2D structure preview, property calculation, group counting, calculation/counting options, and CSV export.

To build a double-clickable Windows app folder:

`python scripts/build_windows_app.py`

The default build output is `dist/Groupy/Groupy.exe`. Use `--mode onefile` to build a single executable.

For a smaller package, build from a clean packaging environment instead of a broad development environment:

```powershell
conda create -n groupy_package -c conda-forge python=3.11 rdkit pandas numpy openpyxl tqdm joblib pyside6 pyinstaller
conda activate groupy_package
python -m pip install -e . --no-deps
python scripts\build_windows_app.py
```

The `_internal` folder contains bundled runtime libraries. Large MKL or BLAS DLLs usually come from the build environment and should not be deleted manually unless the packaged app is retested.

Before distributing a packaged app to ordinary users, follow `RELEASE_CHECKLIST.md`.

The desktop app is intended for ordinary users who need SMILES-based property calculation, group counting, and CSV export. Optional workflows have separate dependency requirements:

- Conversion and Gaussian input generation require OpenBabel from conda-forge and are not part of the default GUI workflow.
- Molecular visualization requires `.[viewer]` and ASE.
- A packaged Windows app should be tested on a clean Windows machine before distribution.
- Package size optimization is intentionally deferred until the user-facing workflow is stable, because removing runtime DLLs without retesting can break the executable.

For Python scripts or GUI integrations, use quiet batch calls:

```python
from groupy.gp_calculator import Calculator
from groupy.gp_counter import Counter

Calculator().calculate_mols("SMILES.txt", "calculate.csv", verbose=False)
Counter().count_mols("SMILES.txt", "count.csv", add_smiles=True, verbose=False)
```

Batch APIs keep processing failed molecules by default for backward compatibility. Pass `continue_on_error=False` when a script or GUI workflow should stop at the first invalid input.

CI runs the smoke tests, source compilation check, `python -m build`, and a dry-run Windows app packaging command.

### Manual and Documention
The user manual is in the manual folder. Generated API documentation can be found in the doc folder and should be kept separate from source documentation.


# Known limitation
when calculating properties of molecules, *simultaneous* type parameters may lead to some mistake results, so we set the 
default parameter type is *stepwise*
