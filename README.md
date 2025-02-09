This repository corresponds to the paper *Groupy: An open-source toolkit for molecular simulation and property calculation*

### Install

Download the source code:

`git clone https://github.com/47-5/Groupy.git`

One may create an environment using Anaconda:

`conda create -n groupy_env python=3.10`

`conda activate groupy_env`

Install:

`pip install .\Groupy\dist\groupy-3.0.0.tar.gz`

`conda install -c conda-forge openbabel`(**Do not** use `pip install openbabel` )

Then one can enter `Groupy` in terminal to start the Groupy.

### Manual and Documention
The user manual is in the manual folder, and the API documentation can be found in the doc folder.


# Sorry!
Sorry, we found a bug, using batch mode to calculate the properties of a batch of molecules results in errors, but the calculation of a single molecule is normal, we recommend users do not use batch mode. We are actively working on this bug
