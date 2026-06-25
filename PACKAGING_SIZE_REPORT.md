# Groupy Packaging Size Report

Created: 2026-06-25

This report records the packaged Windows app size baseline and the result after rebuilding from a clean OpenBLAS-based packaging environment.

## Current Baseline

Measured artifact:

```text
dist/Groupy/Groupy.exe
dist/Groupy/_internal
```

Overall size:

| Path | Size |
| --- | ---: |
| `dist/Groupy` | 772.41 MB |
| `dist/Groupy/_internal` | 753.50 MB |
| `dist/Groupy/Groupy.exe` | 18.91 MB |

Top component groups:

| Component | Count | Size |
| --- | ---: | ---: |
| MKL DLLs | 25 | 538.61 MB |
| ICU DLLs | 12 | 46.96 MB |
| RDKit package and DLLs | 244 | 35.63 MB |
| Qt DLLs | 5 | 22.53 MB |
| PySide6 package | 121 | 22.34 MB |
| pandas package | 53 | 12.50 MB |
| matplotlib package | 211 | 11.22 MB |
| numpy package | 13 | 5.87 MB |
| Tcl/Tk files | 7 | 3.44 MB |

Largest files:

| File | Size |
| --- | ---: |
| `mkl_avx512.3.dll` | 71.92 MB |
| `mkl_core.3.dll` | 68.57 MB |
| `mkl_avx10.3.dll` | 64.55 MB |
| `mkl_avx2.3.dll` | 45.28 MB |
| `mkl_mc3.3.dll` | 44.96 MB |
| `mkl_intel_thread.3.dll` | 37.27 MB |
| `mkl_def.3.dll` | 35.34 MB |
| `icudt78.dll` | 31.58 MB |
| `mkl_tbb_thread.3.dll` | 28.46 MB |
| `mkl_rt.3.dll` | 26.60 MB |

## OpenBLAS Rebuild Result

Measured after rebuilding from a clean conda-forge packaging environment with OpenBLAS-linked numerical dependencies.

Overall size:

| Path | Size |
| --- | ---: |
| `dist/Groupy` | 228.43 MB |
| `dist/Groupy/_internal` | 213.04 MB |
| `dist/Groupy/Groupy.exe` | 15.39 MB |

Top component groups:

| Component | Count | Size |
| --- | ---: | ---: |
| ICU DLLs | 3 | 38.21 MB |
| RDKit package and DLLs | 244 | 35.63 MB |
| OpenBLAS/BLAS DLLs | 3 | 27.32 MB |
| Qt DLLs | 5 | 22.53 MB |
| PySide6 package | 120 | 22.24 MB |
| pandas package | 53 | 12.50 MB |
| numpy package | 13 | 5.87 MB |
| matplotlib package | 0 | 0.00 MB |
| MKL DLLs | 0 | 0.00 MB |

Largest files:

| File | Size |
| --- | ---: |
| `icudt78.dll` | 31.58 MB |
| `openblas.dll` | 27.04 MB |
| `Qt6Gui.dll` | 8.45 MB |
| `libcrypto-3-x64.dll` | 7.08 MB |
| `Qt6Widgets.dll` | 6.24 MB |
| `python311.dll` | 5.91 MB |
| `Qt6Core.dll` | 5.55 MB |

Reduction compared with the MKL-linked baseline:

| Path | Before | After | Reduction |
| --- | ---: | ---: | ---: |
| `dist/Groupy` | 772.41 MB | 228.43 MB | 543.98 MB |
| `dist/Groupy/_internal` | 753.50 MB | 213.04 MB | 540.46 MB |
| `dist/Groupy/Groupy.exe` | 18.91 MB | 15.39 MB | 3.52 MB |

## Interpretation

- MKL is the dominant size source, accounting for about 538.61 MB.
- Qt/PySide6 is required by the desktop GUI and accounts for about 44.87 MB before Qt transitive DLLs such as ICU.
- RDKit is required by property calculation, group counting, and 2D structure preview.
- `matplotlib` appears in the current `_internal` even though the build script excludes it by default. This likely means the measured `dist/Groupy` folder contains stale files from an older build or an over-collected RDKit subtree.
- Rebuilding from a clean OpenBLAS-based packaging environment removed MKL and stale `matplotlib` contents. The current optimized artifact is about 228.43 MB and has been confirmed to run normally by the user.

## Optimization Order

1. Keep using the clean conda-forge packaging environment for release builds.
2. Retest `dist/Groupy/Groupy.exe` after every packaging change.
3. If more reduction is needed, investigate ICU, RDKit, Qt/PySide6, and Tcl/Tk in that order.
4. If RDKit is still over-collected, replace broad `--collect-submodules=rdkit` with a narrower set of required RDKit modules and retest.

## Commands

Recommended clean packaging environment:

```powershell
conda create -n groupy_package -c conda-forge python=3.11 rdkit pandas numpy openpyxl tqdm joblib pyside6 pyinstaller "libblas=*=*openblas"
conda activate groupy_package
python -m pip install -e . --no-deps
python scripts\build_windows_app.py
```

The build script removes `dist/Groupy` before a default onedir build. Use `--no-clean-dist` only when debugging PyInstaller behavior and stale output files are acceptable.
