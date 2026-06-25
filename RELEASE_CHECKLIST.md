# Groupy Release Checklist

Use this checklist before distributing a Windows build to ordinary users.

## 1. Build Environment

- Build from a clean packaging environment, not a broad development environment.
- Prefer conda-forge packages for binary scientific dependencies.
- Install the local project without pulling duplicate dependencies:

```powershell
conda create -n groupy_package -c conda-forge python=3.11 rdkit pandas numpy openpyxl tqdm joblib pyside6 pyinstaller
conda activate groupy_package
python -m pip install -e . --no-deps
```

## 2. Pre-Build Checks

- Run the full smoke-test suite:

```powershell
python -W error::ResourceWarning -m unittest discover -s tests
```

- Confirm the CLI still works:

```powershell
Groupy.exe count --smiles C1CCCC1
Groupy.exe calculate --smiles C1CCCC1
Groupy-GUI.exe --check
```

- Confirm the current release version and user-facing limitations are documented.

## 3. Build The App

- Build the default Windows app folder:

```powershell
python scripts\build_windows_app.py
```

- The expected app entry point is:

```text
dist/Groupy/Groupy.exe
```

- Distribute the complete `dist/Groupy` folder. Do not move only `Groupy.exe` away from `_internal`.
- Do not delete DLLs from `_internal` unless the packaged app is retested afterward.

## 4. GUI Smoke Test

Run `dist/Groupy/Groupy.exe` and check:

- The window opens by double-clicking.
- SMILES file import works for a small `.txt`, `.csv`, or `.xlsx` input.
- The 2D structure preview updates for `C1CCCC1`.
- Property calculation works for `C1CCCC1`.
- Group counting works for `C1CCCC1`.
- Parameter and group-counting options can be changed before running.
- CSV export creates a readable file.
- Invalid SMILES input produces an error row instead of crashing.
- Long-running calculation or counting does not freeze the UI.

## 5. Optional Workflow Boundaries

- The default GUI workflow covers SMILES-based property calculation, group counting, and CSV export.
- OpenBabel conversion and Gaussian input generation require OpenBabel from conda-forge.
- ASE visualization requires the viewer extra and is not part of the default GUI workflow.
- If optional workflows are added to a packaged app later, repeat the clean Windows validation.

## 6. Clean Windows Validation

Test the final artifact on a Windows machine without the development environment:

- Unzip or copy the complete `dist/Groupy` folder.
- Launch `dist/Groupy/Groupy.exe` by double-clicking.
- Repeat the GUI smoke test.
- Test from a path with spaces, such as `C:\Users\<name>\Desktop\Groupy Test`.
- Confirm CSV export works in a user-writable folder.
- Confirm the app still launches after a reboot.

## 7. Release Artifact

- Package the complete `dist/Groupy` folder as a zip archive.
- Include release notes that mention optional dependency boundaries.
- Record the build date, source commit, Python version, and build environment.
- Keep `_internal` size optimization as a separate post-release task until the main workflow is stable.

## 8. Validation Record

- 2026-06-25: User confirmed the packaged app validation can run normally.
