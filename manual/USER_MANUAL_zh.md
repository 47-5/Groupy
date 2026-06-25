# Groupy 用户手册

更新日期：2026-06-25

本手册面向当前重构后的 Groupy，重点介绍普通用户的桌面 GUI、命令行、Python API、文件格式和 Windows 打包发布方式。旧版背景说明和更详细的算法介绍仍保留在 `manual/Groupy_manual.md`。

## 1. Groupy 可以做什么

Groupy 是一个基于 SMILES 的分子分析工具，主要功能包括：

- 根据 SMILES 计算分子性质。
- 根据 SMILES 统计基团贡献方法中的基团数量。
- 将结果导出为 CSV 文件。
- 在桌面 GUI 中查看 SMILES 对应的 2D 分子结构。
- 从 `.txt`、`.csv`、`.xlsx` 文件批量导入 SMILES。
- 作为 Python 包被其他脚本或 GUI 调用。
- 可选地使用 OpenBabel 做格式转换和 Gaussian 输入文件生成。

普通用户推荐使用桌面 GUI；脚本用户和开发者可以使用命令行或 Python API。

## 2. 推荐使用方式

### 2.1 普通用户

如果你拿到的是打包后的 Windows 程序，请进入：

```text
dist/Groupy/Groupy.exe
```

双击 `Groupy.exe` 启动。不要只复制 `Groupy.exe`，必须保留整个 `dist/Groupy` 文件夹，因为 `_internal` 中包含运行所需的 Python、RDKit、Qt、OpenBLAS 等库。

### 2.2 开发者或脚本用户

如果你从源码运行，推荐创建独立环境：

```powershell
conda create -n groupy_env -c conda-forge python=3.11 rdkit pandas numpy openpyxl tqdm joblib
conda activate groupy_env
python -m pip install -e .
```

如果需要 GUI：

```powershell
python -m pip install -e ".[gui]"
```

如果需要打包 Windows 应用：

```powershell
python -m pip install -e ".[gui,package]"
```

如果需要 OpenBabel 相关功能：

```powershell
conda install -c conda-forge openbabel
```

不要使用 `pip install openbabel`，它在 Windows 上通常不可靠。

## 3. 桌面 GUI 使用说明

启动 GUI：

```powershell
Groupy-GUI
```

或者双击打包后的 `dist/Groupy/Groupy.exe`。

### 3.1 输入 SMILES

在左上方 SMILES 输入框中输入一个或多个 SMILES，每行一个。例如：

```text
C1CCCC1
CCO
CC(C)C
```

输入框中第一个有效 SMILES 会显示在右侧 2D 结构预览区域。

### 3.2 从文件导入 SMILES

点击 `Import File` 选择输入文件。支持：

- `.txt`
- `.csv`
- `.xlsx`

`.txt` 文件每行一个 SMILES：

```text
C1CCCC1
CCO
CC(C)C
```

`.csv` 和 `.xlsx` 文件必须包含名为 `smiles` 的列：

```text
smiles
C1CCCC1
CCO
CC(C)C
```

导入后，文件中的 SMILES 会填入输入框。

### 3.3 2D 结构预览

GUI 右侧会显示 SMILES 对应的 2D 结构图。

- 输入框改变时，预览第一个有效 SMILES。
- 计算或统计后，点击结果表中的某一行，预览会切换到该行对应的分子。
- 如果 SMILES 无法解析，预览区会显示 `Invalid SMILES`。

该功能使用 RDKit 生成 2D 图，不需要 OpenBabel 或 ASE。

### 3.4 计算分子性质

点击 `Calculate Properties` 计算性质。结果会显示在表格中，并可导出为 CSV。

计算选项：

- `Parameters`
  - `Stepwise`：默认推荐选项。
  - `Simultaneous`：保留为可选参数类型；当前已知可能在部分分子上产生不可靠结果。
- `Hydrocarbon check`
  - 勾选时，燃烧焓、热值、比冲等仅在烃类分子上计算。
  - 取消勾选时，会尝试对非烃类分子也计算这些性质，但结果可能没有物理意义。

### 3.5 统计基团数量

点击 `Count Groups` 统计基团数量。

统计选项：

- `Show zero groups`
  - 勾选后，结果中会显示所有基团，包括数量为 0 的基团。
  - 不勾选时，只显示非零基团。
- `Include SMILES`
  - 勾选后，结果中包含原始或规范化后的 SMILES 列。

### 3.6 导出 CSV

计算或统计完成后，点击 `Export CSV` 保存结果。

CSV 文件可用 Excel、WPS、LibreOffice 或 Python/pandas 打开。

### 3.7 GUI 常见问题

如果从源码运行 GUI 时出现 PySide6 缺失提示：

```powershell
python -m pip install -e ".[gui]"
```

如果双击打包后的 exe 无反应：

- 确认没有只复制 `Groupy.exe`。
- 确认 `_internal` 文件夹仍在 `Groupy.exe` 同级目录。
- 尝试把整个 `dist/Groupy` 文件夹复制到没有中文或特殊字符的路径下测试。
- 从命令行运行 `dist\Groupy\Groupy.exe` 查看错误信息。

## 4. 命令行使用

安装后会提供 `Groupy` 命令。

### 4.1 查看帮助

```powershell
Groupy --help
Groupy count --help
Groupy calculate --help
Groupy convert --help
```

### 4.2 统计单个 SMILES 的基团

```powershell
Groupy count --smiles C1CCCC1
```

输出为 JSON，例如：

```json
{"f_168": 5, "smiles": "C1CCCC1"}
```

导出 CSV：

```powershell
Groupy count --smiles C1CCCC1 --output count.csv
```

显示零值基团：

```powershell
Groupy count --smiles C1CCCC1 --include-zero --output count_full.csv
```

### 4.3 计算单个 SMILES 的性质

```powershell
Groupy calculate --smiles C1CCCC1
```

导出 CSV：

```powershell
Groupy calculate --smiles C1CCCC1 --output calculate.csv
```

指定参数类型：

```powershell
Groupy calculate --smiles C1CCCC1 --parameter-type step_wise
Groupy calculate --smiles C1CCCC1 --parameter-type simultaneous
```

取消烃类检查：

```powershell
Groupy calculate --smiles CCO --no-check-hydrocarbon
```

### 4.4 批量计算

输入文件可以是 `.txt`、`.csv` 或 `.xlsx`。

```powershell
Groupy calculate --input SMILES.txt --output calculate.csv
Groupy count --input SMILES.txt --output count.csv
```

`.csv` 和 `.xlsx` 文件必须包含 `smiles` 列。

### 4.5 文件格式转换

单文件转换命令：

```powershell
Groupy convert --input molecule.xyz --from xyz --to mol2 --output molecule.mol2
```

该功能需要 OpenBabel：

```powershell
conda install -c conda-forge openbabel
```

转换功能不是默认 GUI 主流程的一部分。

### 4.6 旧交互菜单

直接运行：

```powershell
Groupy
```

会进入旧版交互菜单。新脚本和自动化流程建议优先使用：

- `Groupy count`
- `Groupy calculate`
- `Groupy convert`
- `groupy.api`

## 5. Python API

Groupy 可以作为 Python 包使用。

### 5.1 单分子计算

```python
from groupy.api import calculate_smiles, count_smiles

properties = calculate_smiles("C1CCCC1")
groups = count_smiles("C1CCCC1")

print(properties)
print(groups)
```

### 5.2 批量计算

```python
from groupy.api import calculate_many_smiles, count_many_smiles, write_records_csv

smiles_values = ["C1CCCC1", "CCO", "CC(C)C"]

properties = calculate_many_smiles(smiles_values)
groups = count_many_smiles(smiles_values)

write_records_csv(properties, "calculate.csv")
write_records_csv(groups, "count.csv")
```

### 5.3 从文件读取 SMILES

```python
from groupy.io import load_smiles_file

smiles_values = load_smiles_file("SMILES.xlsx")
```

`.csv` 和 `.xlsx` 文件必须包含 `smiles` 列。

### 5.4 使用底层类

如果需要旧 API 或更细控制，可以直接使用底层类：

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

默认情况下，批处理遇到坏 SMILES 会继续执行并记录失败信息。如果希望遇到第一个错误就停止：

```python
calculator.calculate_mols(
    "SMILES.txt",
    properties_file_path="calculate.csv",
    continue_on_error=False,
    verbose=False,
)
```

## 6. 可选进阶功能

### 6.1 OpenBabel 转换

OpenBabel 相关功能包括：

- SMILES 生成 `.xyz`。
- 常见分子文件格式转换。
- 从结构文件提取 SMILES。
- Gaussian 输入文件生成中的结构转换步骤。

安装：

```powershell
conda install -c conda-forge openbabel
```

### 6.2 Gaussian 输入文件生成

`groupy.gp_generator.Generator` 可根据 SMILES 生成 Gaussian `.gjf` 文件。该流程通常依赖 OpenBabel 生成三维坐标。

这部分目前属于进阶脚本功能，不是默认 GUI 主流程。

### 6.3 分子可视化

旧版 Viewer 依赖 ASE：

```powershell
python -m pip install -e ".[viewer]"
```

或：

```powershell
conda install -c conda-forge ase
```

当前 GUI 的 2D 结构预览不依赖 ASE。

## 7. 打包 Windows 应用

如果需要给普通用户发布双击运行的 Windows 应用，推荐使用干净的 conda-forge + OpenBLAS 打包环境：

```powershell
conda create -n groupy_package -c conda-forge python=3.11 rdkit pandas numpy openpyxl tqdm joblib pyside6 pyinstaller "libblas=*=*openblas"
conda activate groupy_package
python -m pip install -e . --no-deps
python scripts\build_windows_app.py
```

默认输出：

```text
dist/Groupy/Groupy.exe
```

发布时请压缩并分发整个 `dist/Groupy` 文件夹，不要只分发 exe。

打包脚本默认会删除旧的 `dist/Groupy` 输出目录，以避免旧 `_internal` 文件残留影响体积。只有调试 PyInstaller 时才建议使用：

```powershell
python scripts\build_windows_app.py --no-clean-dist
```

体积优化记录见：

```text
PACKAGING_SIZE_REPORT.md
```

发布前检查清单见：

```text
RELEASE_CHECKLIST.md
```

## 8. 已知限制

- `simultaneous` 参数类型在部分分子上可能产生不可靠结果，默认建议使用 `step_wise`。
- 燃烧焓、热值、比冲等性质主要针对烃类分子设计；非烃分子的结果需要谨慎解释。
- OpenBabel 转换、Gaussian 输入文件生成、ASE 可视化属于可选进阶功能，不是默认 GUI 主流程。
- 打包后的 `_internal` 是运行时依赖目录，不应手动删除其中 DLL，除非重新完整测试 exe。
- 进一步的 GUI 高级页面，例如 OpenBabel 转换和 Gaussian 输入生成，可以作为后续阶段开发。

## 9. 故障排查

### 9.1 无法解析 SMILES

检查：

- 是否有空格或不可见字符。
- 环闭合数字是否成对。
- 原子符号大小写是否正确。
- 芳香原子是否使用正确小写形式。

### 9.2 CSV 或 XLSX 无法导入

检查文件中是否有名为 `smiles` 的列。列名区分大小写，建议使用小写：

```text
smiles
```

### 9.3 打包后体积过大

优先确认是否使用了 OpenBLAS 打包环境：

```powershell
conda create -n groupy_package -c conda-forge python=3.11 rdkit pandas numpy openpyxl tqdm joblib pyside6 pyinstaller "libblas=*=*openblas"
```

如果 `_internal` 中存在大量 `mkl_*.dll`，说明构建环境仍然链接了 Intel MKL。

### 9.4 GUI 启动失败

源码运行时检查：

```powershell
Groupy-GUI --check
```

如果提示缺少 PySide6：

```powershell
python -m pip install -e ".[gui]"
```

打包应用启动失败时，优先确认整个 `dist/Groupy` 文件夹是否完整。
