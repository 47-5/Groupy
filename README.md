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

To launch the desktop GUI after installing GUI support:

`Groupy-GUI`

To build a double-clickable Windows app folder:

`python scripts/build_windows_app.py`

The default build output is `dist/Groupy/Groupy.exe`. Use `--mode onefile` to build a single executable.

For Python scripts or GUI integrations, use quiet batch calls:

```python
from groupy.gp_calculator import Calculator
from groupy.gp_counter import Counter

Calculator().calculate_mols("SMILES.txt", "calculate.csv", verbose=False)
Counter().count_mols("SMILES.txt", "count.csv", add_smiles=True, verbose=False)
```

### Manual and Documention
The user manual is in the manual folder, and the API documentation can be found in the doc folder.


# Known limitation
when calculating properties of molecules, *simultaneous* type parameters may lead to some mistake results, so we set the 
default parameter type is *stepwise*
