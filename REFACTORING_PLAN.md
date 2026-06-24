# Groupy Refactoring Plan

Created: 2026-06-24

This document captures the agreed refactoring direction for Groupy. The guiding principle is to stabilize behavior first, then modernize structure in small, verifiable steps.

## Goals

- Make local development work with `python -m pip install -e .`.
- Keep existing scientific behavior stable while changing project structure.
- Separate library APIs from command-line interaction.
- Prepare for a desktop GUI that ordinary users can launch by double-clicking.
- Improve maintainability without rewriting the chemistry logic all at once.
- Build a minimal regression test suite before deeper refactors.

## GUI Target

The final user-facing application should be a desktop GUI, not a web dashboard. The preferred route is PySide6/Qt because it supports a native-feeling Windows application and can be packaged into a double-clickable executable.

Recommended GUI scope:

- Single-molecule property calculation from SMILES.
- Single-molecule group counting from SMILES.
- Batch calculation from `.txt`, `.csv`, or `.xlsx`.
- Batch group counting from `.txt`, `.csv`, or `.xlsx`.
- CSV export for tabular results.
- Clear dependency warnings for optional conversion/viewer features.

Packaging target:

- Windows executable or application folder built with PyInstaller or Nuitka.
- No requirement for end users to open a terminal.
- OpenBabel-dependent features should be optional or packaged only after dependency behavior is well understood.

The GUI should call stable library APIs, not the legacy `input()` menu.

## Phase 1: Packaging And Repository Hygiene

- [x] Keep `pyproject.toml` as the canonical packaging metadata.
- [x] Keep `setup.py` only as a legacy compatibility shim, or remove it once no longer needed.
- [x] Remove generated artifacts from Git tracking:
  - `build/`
  - `dist/`
  - `Groupy.egg-info/`
- [x] Confirm `.gitignore` continues to ignore those generated paths.
- [x] Verify:
  - `python -m pip install -e .`
  - `python -m pip show Groupy`
  - `python -m compileall groupy`

## Phase 2: Regression Test Baseline

Add a small smoke-test suite before changing core behavior.

- [x] Add a test baseline in `tests/test_smoke.py`.
- [x] Add `pytest` to the `dev` optional dependency group.
- [x] Test that `Loader` can load bundled Excel data.
- [x] Test that `Counter().count_a_mol("C1CCCC1")` returns a stable result.
- [x] Test that `Calculator().calculate_a_mol("C1CCCC1")` completes and returns expected keys.
- [x] Test `Tool.load_smiles_iterator()` for:
  - `.txt`
  - `.csv`
  - `.xlsx`
- [x] Test that the console entry point starts and can exit cleanly.

Current test command:

```powershell
python -W error::ResourceWarning -m unittest discover -s tests
```

## Phase 3: Separate Core API From CLI

The current `groupy_main.py` is a fully interactive menu. Preserve it initially, but introduce a modern CLI layer around the existing API.

- [x] Add `groupy/cli.py`.
- [x] Add `groupy/api.py` for non-interactive workflows shared by CLI and future GUI code.
- [x] Move console entry point from `groupy.groupy_main:main` to the new CLI once ready.
- [x] Preserve the old interactive menu as a subcommand or compatibility path.
- [ ] Add scriptable commands such as:
  - [x] `Groupy count --smiles C1CCCC1`
  - [x] `Groupy calculate --smiles C1CCCC1`
  - [x] `Groupy calculate --input SMILES.txt --output result.csv`
  - `groupy convert --input molecule.xyz --from xyz --to mol2 --output molecule.mol2`
- [x] Keep command-line parsing separate from chemistry logic.

Current CLI behavior:

- `Groupy` starts the legacy interactive menu.
- `Groupy interactive` starts the legacy interactive menu explicitly.
- `Groupy count --smiles C1CCCC1` prints nonzero group counts as JSON.
- `Groupy count --smiles C1CCCC1 --output count.csv` writes one-row CSV output.
- `Groupy calculate --smiles C1CCCC1` prints calculated properties as JSON.
- `Groupy calculate --input SMILES.txt --output calculate.csv` writes batch calculated properties to CSV.

## Phase 4: File And Path Handling

- [ ] Replace scattered `os.path` logic with `pathlib.Path` where practical.
- [x] Centralize SMILES file loading into one helper.
- [x] Replace `list(open(...))` with context-managed file reads for SMILES input.
- [x] Use explicit encodings for text SMILES files.
- [ ] Avoid writing implicit side-effect files such as `error.txt`, `xyz_fail.txt`, and `gjf_fail.txt` to the current working directory unless requested.
  - [x] `Calculator.calculate_mols()` no longer writes `error.txt` unless `error_file_path` is provided.
  - [x] `Convertor.batch_smi_to_xyz()` no longer writes `xyz_fail.txt` or `xyz_succeed.txt` unless explicit paths are provided.
  - [x] `Generator.batch_smi_to_gjf()` no longer writes `gjf_fail.txt` or `gjf_succeed.txt` unless explicit paths are provided.
- [ ] Make batch output paths predictable and configurable.

Current file handling status:

- `groupy.io.load_smiles_file()` is the shared loader for `.txt`, `.csv`, and `.xlsx` SMILES inputs.
- `groupy.api`, `groupy.gp_tool.Tool`, `Calculator.calculate_mols*`, and `Counter.count_mols*` use the shared loader.
- `Calculator.calculate_mols()` now supports an explicit `error_file_path` for failed SMILES output.
- `Convertor.batch_smi_to_xyz()` now supports explicit `fail_file_path` and `succeed_file_path` outputs.
- `Generator.batch_smi_to_gjf()` now supports explicit `fail_file_path` and `succeed_file_path` outputs.

## Phase 5: Exceptions, Logging, And Error Reporting

- [ ] Replace bare `except:` blocks with specific exceptions.
- [ ] Replace `raise NotImplemented(...)` with `NotImplementedError` or `ValueError`.
- [ ] Move user-facing `print()` calls toward the CLI layer.
- [ ] Use `logging` in library code.
- [ ] For batch processing, continue processing failed molecules only when configured to do so.
- [ ] Preserve enough failure detail to debug invalid SMILES, unsupported formats, and dependency problems.

## Phase 6: Dependency Boundaries And Lazy Imports

- [x] Keep core counting and property calculation independent from OpenBabel.
- [x] Lazy-load OpenBabel only inside conversion functions that need it.
- [x] Lazy-load visualization dependencies only inside viewer functions that need them.
- [x] Consider optional dependency groups:
  - `.[convert]`
  - `.[viewer]`
  - `.[dev]`
- [x] Document that OpenBabel should usually be installed from conda-forge.

Current dependency boundary status:

- `groupy.gp_convertor` and `groupy.gp_generator` can be imported without importing OpenBabel.
- OpenBabel is loaded through `groupy.gp_convertor._load_pybel()` only when a conversion feature needs Pybel.
- Missing OpenBabel now raises an install hint for the relevant conversion feature.
- `groupy.gp_viewer` can be imported without importing ASE.
- ASE is loaded through viewer helpers only when molecular visualization needs it.
- Core installation no longer depends on ASE or OpenBabel.
- `.[viewer]` installs ASE for molecular visualization workflows.
- `.[convert]` is intentionally empty because OpenBabel should usually be installed from conda-forge.
- `.[dev]` installs build and test tooling.

## Phase 7: Data Loading And Performance

- [x] Cache `Loader` results so Excel files are not repeatedly parsed.
- [x] Add regression coverage around cached loader reads.
- [ ] Keep SMARTS/group-counting logic behaviorally unchanged until tests cover it better.
- [x] Rename or alias `*_mpi` APIs to `*_parallel`, since they use `joblib`, not MPI.
- [x] Preserve old method names during transition for backward compatibility.

Current data loading status:

- `Loader.load_parameters()` caches bundled Excel parameter tables per parameter type.
- `Loader.load_group_order()` caches bundled group order tables.
- `Loader.clear_cache()` is available for tests and advanced workflows that need to refresh cached internal data.
- Public loader methods return independent copies so callers cannot mutate the process-wide cache by accident.

Current parallel API status:

- `Calculator.calculate_mols_parallel()` calls `calculate_mols_mpi()`.
- `Counter.count_mols_parallel()` calls `count_mols_mpi()`.
- `Convertor.batch_smi_to_xyz_parallel()` calls `batch_smi_to_xyz_mpi()`.
- `Convertor.batch_convert_file_type_parallel()` calls `batch_convert_file_type_mpi()`.
- `Convertor.batch_file_to_smi_parallel()` calls `batch_file_to_smi_mpi()`.
- `Generator.batch_smi_to_gjf_parallel()` calls `batch_smi_to_gjf_mpi()`.

## Phase 8: Documentation And CI

- [ ] Update README with:
  - [x] editable install
  - [x] conda/OpenBabel note
  - [ ] library API examples
  - [x] CLI examples
- [ ] Add a build check:
  - `python -m build`
- [ ] Add GitHub Actions for:
  - install
  - lint
  - tests
  - package build
- [ ] Keep generated API documentation separate from source documentation.

## Phase 9: Desktop GUI And Application Packaging

- [ ] Choose GUI framework, currently recommended: PySide6.
- [ ] Add a minimal `groupy/gui/` package.
- [ ] Build first GUI screen for:
  - SMILES input
  - property calculation
  - group counting
  - CSV export
- [ ] Add background worker handling so long calculations do not freeze the UI.
- [ ] Add GUI smoke tests where practical.
- [ ] Add packaging script for Windows executable builds.
- [ ] Test the packaged app on a clean Windows environment.
- [ ] Document limitations around OpenBabel and optional conversion features.

## Current Known Review Findings

- `Groupy.egg-info/`, `build/`, and `dist/` are tracked generated artifacts.
- The command-line interface is tightly coupled to `input()` prompts.
- Importing the current CLI path can require OpenBabel even for workflows that do not use conversion.
- Several modules use bare `except:` and swallow useful debugging information.
- Some file reads use `list(open(...))` without context managers or explicit encoding.
- Batch workflows write failure logs to fixed filenames in the current working directory.

## Near-Term Next Step

Continue with Phase 7:

1. Keep SMARTS/group-counting behavior unchanged while adding narrower tests around representative groups.
2. Start replacing broad exception handling in batch paths with explicit errors and logging.
3. After the core API is stable enough, begin the minimal PySide6 GUI shell.
