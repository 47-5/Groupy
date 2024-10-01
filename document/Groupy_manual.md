# Groupy 3.0 Manual

[toc]

## 1 Overview

Groupy is a program for calculating various molecular properties and preparing input files of molecular simulation software such as Gaussian. This code requires only SMILES as input, but can output many new useful data and files in multiple formats. The output information is clear and easy to read. The tips to the users are very detailed and easy to follow when using. Message passing interface (MPI) parallelization is supported to reduce computing time when the properties of a large number of molecules are calculated. Groupy not only supports the calculation of molecular properties using the traditional group contribution method, but also directly outputs the group-contribution-style molecular fingerprints for machine learning. The code has strong extensibility, which can be used as an external library to build other programs. We hope that Groupy brings great convenience to both computational and experimental chemists in their daily research. The code of Groupy can be freely obtained at https://github.com/47-5/Groupy



### 1.1 Background

This section introduces the physical and chemical background and computer background of the development of this program.



#### 1.1.1 Group contribution method

Since Macleod introduced the group contribution method for calculating the  molar volume of liquids in 1923, the group contribution method has  undergone rapid development over the past six decades. This method is a  highly accurate approach for estimating the physicochemical properties  of compounds. It offers advantages such as simplicity in prediction  processes, wide applicability, and good generality. Since its proposal  in the mid-20th century, this method has been widely utilized for  estimating the physical properties of pure substances. It is also  employed in equilibrium calculations between different phases,  particularly in estimating equilibria between gas and liquid phases. The group contribution method is a means for predicting phase  equilibrium, enabling not only the estimation and prediction of the  physical and thermodynamic properties of pure substances but also the  prediction of the thermodynamic properties of mixtures. Presently, the  group contribution method can be used to predict various physical  properties of pure substances (such as critical parameters, molar  volumes, refractive indices, thermal conductivities, viscosities, molar  volumes) as well as various thermodynamic properties of pure substances  (including heat capacities, heat of vaporization, saturated vapor  pressures, standard enthalpies of formation), and even the thermodynamic properties of mixtures (such as activity coefficients).

The fundamental assumption of the group contribution method is that  the physical properties of a pure compound or mixture are equivalent to  the sum of the contributions of various groups that constitute the  compound or mixture. In other words, it assumes that the contribution of the same group to physical properties is consistent across different  systems. The key advantage of the group contribution method lies in its  high level of generality. While the number of molecules in the material  world is vast and challenging to count, the number of common organic  groups that constitute organic compounds is relatively small, typically  in the hundreds. Therefore, by leveraging existing experimental values  to estimate the contributions of different groups to various physical  properties, it is possible to predict the properties of other organic  compounds.

The Groupy implements the third-order group  contribution method proposed in "Group-contribution based estimation of  pure component properties," incorporating additional groups and  parameters introduced in "Group-contribution+ (GC+) based estimation of  properties of pure components: Improved property estimation and  uncertainty analysis."

#### 1.1.2 SMILES

SMILES (Simplified Molecular Input Line Entry System) is a line notation (a typographical method using printable characters) for entering and representing molecules and reactions, which was designed by Daylight. SMILES can succinctly represent the topology of a given molecule, while the hydrogen atoms in the molecule are omitted. SMILES, as a system that can fully describe molecular topology, is widely used in the establishment of databases and the training of generative deep learning models.

Below are some rules for SMILES notation:

1. Atoms are represented by their atomic symbols: C for carbon, O for oxygen, N for nitrogen, etc.
2. Hydrogen atoms are often omitted, and implicit hydrogen atoms are assumed based on the valency of the atom.
3. Single bonds are assumed between atoms unless specified otherwise.
4. Rings are indicated by adding a number after an atom to show closure in a cyclic structure.
5. Branches are enclosed in parentheses.
6. Aromatic rings are represented by lowercase letters (e.g., c for benzene).
7. Double and triple bonds are specified by using '=' and '#' symbols, respectively.
8. Chirality can be denoted using '@' symbols.
9. Isotopes can be indicated by adding a mass number before the atomic symbol (e.g., [13C] for carbon-13).
10. Aromaticity in a ring can be represented using lowercase letters or by explicitly using the aromatic bond symbol ':'.

These rules provide a basic framework for encoding molecular structures using SMILES notation.


### 1.2 Dependency

Groupy relies on the following python packages：

+ python >= 3.6
+ tqdm
+ numpy
+ pandas
+ ase
+ rdkit 
+ openbabel
+ joblib

The older or newer versions of the aforementioned dependent packages are unlikely to pose significant issues; however, further extensive testing has not been conducted. Testing has been performed on Windows 11,  Ubuntu 18.04, and CentOS 7.8, where no anomalies were detected.

### 1.3 Architecture of Groupy code

Groupy consists of 6 modules, namely Loader, Counter, Calculator, Convertor, Viewer and Generator. The architecture is illustrated in Figure 1, which overviews the modules in Groupy. Loader is responsible for loading internal data of Groupy, i.e. model parameters and other hyperparameters of the group contribution method. Counter is responsible for counting the number of different types of groups in a given molecule, and can also output a group-contribution style molecular fingerprint that can be used for machine learning. Calculator receives the results of Loader and Counter to calculate different properties of the molecule. Convertor implements the conversion of SMILES to common structural files such as gro, xyz, POSCAR, as well as the conversion of one format to another among the common chemical files. Viewer provides an interface with ASE to visualize three-dimensional molecular structures. Generator can generate input files of commonly used computational chemistry software according to the molecular structure, such as single point energy calculation,  geometric optimization, and frequency analysis of Gaussian. The computational chemistry software supported by Generator will be constantly updated.

![](.\figure\Architecture.png)

<div align = "center">Figure1. Architecture and the modular map of Groupy code. The direction of the arrows indicates the direction of the data, the rounded gray boxes represents the module in Groupy, and the white boxes represents the input or output data. </div>

### 1.4 Install

Download the source code:

`git clone https://github.com/47-5/Groupy.git`

One may create an environment using Anaconda:

`conda create -n groupy_env python=3.10`

`conda activate groupy_env`

Install:

`pip install .\Groupy\dist\groupy-3.0.0.tar.gz`

`conda install -c conda-forge openbabel`(**Do not** use `pip install openbabel` )

Then one can enter `Groupy` in terminal to start the Groupy.

## 2 Quick start

This section describes how to use the program to calculate the physical and chemical properties of a molecule and to count the number of groups of a given molecule, as well as some other main functions.

### 2.1 Input and input file

Groupy supports two main modes of operation, one that computes a single molecule and the other that computes all molecules recorded in a given file in batches.

#### 2.1.1 Single molecule

When computing properties for an individual molecule, one can directly  provide the SMILES of the molecule. For instance, the SMILES of cyclohexane is `C1CCCC1`.

#### 2.1.2 分子文件

A file recording some SMILES of molecules should be provided when one want to calculate properties of a batch of molecules.

Its format can be written as follows:

##### 2.1.2.1 txt

```
CCCCCCCC
CCCC(C)C
CCC(C)(C)C
C=CCCCC
C/C=C/CCC
C=C(C)CC
CC=C(C)C
CC(C)=C(C)C
C=C=CC
C=C=C(C)C
```

There should be only one molecule's SMILES per line (no spaces), and no blank lines.

##### 2.1.2.2 csv

```shell
index,smiles,molar_mass,
0,CC,30.069999999999993,
1,C1CC1,42.08100000000002,
2,CCC,44.09700000000002, 
3,C1CCC1,56.10800000000002, 
```

The format is free, as long as there is a column named `smiles`, the rest of the data will be ignored, but there should be no blank lines.

##### 2.1.2.3 xlsx

| index | smiles | molar_mass |
| ----- | ------ | ---------- |
| 0     | CC     | 30.07      |
| 1     | C1CC1  | 42.081     |
| 2     | CCC    | 44.097     |
| 3     | C1CCC1 | 56.108     |
| 4     | CC1CC1 | 56.108     |

The format is free, as long as there is a column named `smiles`, the rest of the data will be ignored, but there should be no blank lines.

### 2.2 Run Groupy

When utilizing this program, users can employ Groupy as a standalone  application. Additionally, to retain the maximum extensibility of Python itself, users can import this program as an external library into  Python scripts they create themselves.

The two distinct ways of utilizing Groupy are outlined below.

#### 2.2.1 Groupy as a standalone program

**Note: The system I used when writing this manual is Windows 11, and the terminal is Anaconda Powershell Prompt provided by Anaconda, which supports some common commands in Linux and supports both Linux and Windows path formats. Please distinguish the file path formats of different systems when using it! **

After the installation is complete according to the instructions in Section 1.4, enter `Groupy` in the terminal to start Groupy. The user will see the main interface as shown in Figure 2.1:

![](.\figure\main_interface.png)

<div align = "center">Figure 2.1 Main interface of Groupy</div>

The program first displays a basic information, including the program name, developer, and developer contact information (users can contact the developer if they encounter any problems during use, and the developer will provide as much help as possible within their capabilities). Then the user's location (main interface) is displayed, and the program asks the user what operation to perform. The user only needs to enter the corresponding serial number to command the program to perform the corresponding task.

##### 2.2.1.1 退出(q)

To exit the program gracefully, just enter `q` in the main interface and press `Enter` on the keyboard, as shown in Figure 2.2.

![](.\figure\main_q.png)

<div align = "center">Figure 2.2 Exit of Groupy</div>

##### 2.2.1.2 Visualizing molecules (0)



##### 2.2.1.2 计算单个分子的理化性质(1)

若要计算单个分子的理化性质，只需在主界面内输入`1`，然后在键盘上敲击`Enter`，根据后续的提示输入所需指令即可，如图2.3所示。

![](C:\Users\lrc\Desktop\group_contribution_3.2_dev\document\figure\main_1.png)

<div align = "center">图2.3 GC程序的主功能1</div>

当通过在主界面输入`1`进入主功能1（计算单个分子的理化性质后），程序会提示用户键入目标分子的SMILES表达式。当用户输入要计算的分子的SMILES（如环戊烷C1CCCC1）并敲击`Enter`后，程序开始进行计算，然后将计算结果输出到屏幕上。之后程序会询问用户是否需要将此结果导出为一个`.csv`文件，若用户确实想要导出，则键入`y`并敲击`Enter`，一个名为`{当前计算的分子的SMILES}_calculate.csv`的文件将产生在程序的主目录下。若用户不需将结果输出为文件，键入`n`并敲击`Enter`即可。

之后程序会再次返回主界面，此时若键入`q`并敲击`Enter`即可优雅地退出程序。



##### 2.2.1.3 统计单个分子的基团数目(2)

若要统计单个分子的基团数目，只需在主界面内输入`2`，然后在键盘上敲击`Enter`，根据后续的提示输入所需指令即可，如图2.4所示。

![](C:\Users\lrc\Desktop\group_contribution_3.2_dev\document\figure\main_2.png)

<div align = "center">图2.4 GC程序的主功能2，使用清爽模式输出结果</div>

当通过在主界面输入`2`进入主功能2（统计单个分子的基团数目后），程序会提示用户键入目标分子的SMILES表达式。当用户输入要计算的分子的SMILES（如环戊烷C1CCCC1）并敲击`Enter`后，程序会询问用户是否使用清爽模式，清爽模式是指仅输出数目不为零的基团结果，而那些数目为零的基团则不会输出其结果（因为已经是0了）。当用户想要以清爽模式输出统计结果时，键入`y`然后敲击`Enter`，程序就开始进行统计，然后将统计结果输出到屏幕上。统计完成后，程序会询问用户是否需要将此结果导出为一个`.csv`文件，若用户确实想要导出，则键入`y`并敲击`Enter`，一个名为`{当前计算的分子的SMILES}_count.csv`的文件将产生在程序的主目录下。若用户不需将结果输出为文件，键入`n`并敲击`Enter`即可。

之后程序会再次返回主界面，此时若键入`q`并敲击`Enter`即可优雅地退出程序。

若用户希望将全部统计结果都输出，则不应使用清爽模式（这在希望获取基团贡献法风格的分子指纹时很常见），此时用户的输入和程序的输出见图2.5。

![](C:\Users\lrc\Desktop\group_contribution_3.2_dev\document\figure\main_2_no_clearmode.png)

<div align = "center">图2.5 GC程序的主功能2，不使用清爽模式输出结果</div>



##### 2.2.1.4 批量计算分子的理化性质(3)

若用户需要计算一批分子的理化性质，需要先准备一个2.1.2小节中介绍的分子文件，然后在主界面内输入`3`，然后在键盘上敲击`Enter`，根据后续的提示输入所需指令即可，如图2.6所示。

![](C:\Users\lrc\Desktop\group_contribution_3.2_dev\document\figure\main_3.png)

<div align = "center">图2.6 GC程序的主功能3</div>

当通过在主界面输入`3`进入主功能3（批量计算给定分子文件中的分子的理化性质），程序会提示用户键入保存了分子SMILES的文件路径。用户输入文件路径（特别提醒：Windows和Linux下路径的格式略有不同！分子文件中不要有空行！）然后敲击`Enter`后，程序会自动读取分子文件中记录的SMILES，然后进行计算。当计算完成后，结果将写进程序主目录下的`batch_results.csv`（自动生成），其格式与2.2.1.2中导出的文件是一样的。



##### 2.2.1.5 基于MPI并行批量计算分子的理化性质(-3)

现代计算机的中央处理器（CPU）往往是多核的，若在批量计算分子性质时希望充分利用CPU的多核性能，可以使用MPI并计算。

若用户需要并行计算一批分子的理化性质，同2.2.1.4小节中一样，需要先准备一个2.1.2小节中介绍的分子文件，然后在主界面内输入`-3`，然后在键盘上敲击`Enter`，根据后续的提示输入所需指令即可，如图2.7所示。

![](C:\Users\lrc\Desktop\group_contribution_3.2_dev\document\figure\main_mins3.png)

<div align = "center">图2.7 GC程序的主功能-3</div>

当通过在主界面输入`-3`进入主功能-3（批量计算给定分子文件中的分子的理化性质），程序会提示用户键入保存了分子SMILES的文件路径。用户输入文件路径（特别提醒：Windows和Linux下路径的格式略有不同！分子文件中不要有空行！）然后敲击`Enter`后，程序会询问用户需要使用多少个进程（核）来并行计算，当用户输入想要调用的核数并敲击`Enter`后，程序会给出提示：**请键入q以优雅地退出程序，然后在终端输入下列命令**，用户需要根据所使用的系统选择输入哪条命令，以Windows为例，在终端输入`mpiexec -np 4 python .\gp_3x_mpirun.py -smiles_file_path ./gp_3x_test_mol/SMILES.txt -result_file_path mpi_batch_calculate_results.csv -task calculate`（**这里的命令只是例子，根据分子文件路径和想要调用的核数不同，屏幕上输出的命令也会有所不同，请根据实际情况随机应变**），会调用本程序中的`gp_3x_mpirun.py`模块进行并行计算，然后将结果写入生成在程序主目录的`mpi_batch_calculate_results.csv`中，其格式与2.2.1.2中导出的文件是一样的。



**注意：基于MPI并行批量计算分子的理化性质的速度优势仅在要计算的分子非常多时（如上万）才能体现，这是因为MPI多进程计算时进程之间相互通信会增加耗时，若要计算的分子数目不多（如几百上千），那么并行计算所减少的耗时无法抵消进程间相互通信所增加的耗时。更具体的例子，可以参考表2.1**

我们在 13th Gen Intel(R) Core(TM) i9-13900KF上对并行效率进行了测试，输入的分子文件使用的是Group Contribution内部数据文件夹中的`gdb.txt`(记录了大约三十万个饱和碳氢分子的SMILES)。结果如下：

<div align = "center">表2.1 GC程序主功能4的并行效率测试</div>

|  并行条件   | 耗时/s |
| :---------: | :----: |
| single core |  1471  |
| mpi-4 core  |  1030  |
| mpi-8 core  |  581   |
| mpi-16 core |  351   |



##### 2.2.1.6 批量统计分子的基团数目(4)

若用户需要统计一批分子的基团数目（这在用户希望使用基团贡献法风格的分子指纹作为下游机器学习模型的输入时非常有用），需要先准备一个2.1.2小节中介绍的分子文件，然后在主界面内输入`4`，然后在键盘上敲击`Enter`，根据后续的提示输入所需指令即可，如图2.8所示。

![](C:\Users\lrc\Desktop\group_contribution_3.2_dev\document\figure\main_4.png)

<div align = "center">图2.8 GC程序的主功能4</div>

当通过在主界面输入`4`进入主功能4（批量统计给定分子文件中的分子的基团数目），程序会提示用户键入保存了分子SMILES的文件路径。用户输入文件路径（特别提醒：Windows和Linux下路径的格式略有不同！分子文件中不要有空行！）然后敲击`Enter`后，程序会自动读取分子文件中记录的SMILES，然后进行计算。当计算完成后，结果将写进程序主目录下的`batch_count_result.csv`（自动生成），其格式与2.2.1.3中导出的文件是一样的。



##### 2.2.1.7 基于MPI并行批量统计分子的基团数目(-4)

现代计算机的中央处理器（CPU）往往是多核的，若在批量统计分子基团数目时希望充分利用CPU的性能，可以使用MPI并计算。

若用户需要统计一批分子的基团数目（这在用户希望使用基团贡献法风格的分子指纹作为下游机器学习模型的输入时非常有用），需要先准备一个2.1.2小节中介绍的分子文件，然后在主界面内输入`-4`，然后在键盘上敲击`Enter`，根据后续的提示输入所需指令即可，如图2.9所示。

![](C:\Users\lrc\Desktop\group_contribution_3.2_dev\document\figure\main_mins4.png)

<div align = "center">图2.9 GC程序的主功能-4</div>

当通过在主界面输入`-4`进入主功能-4（批量统计给定分子文件中的分子的基团数目），程序会提示用户键入保存了分子SMILES的文件路径。用户输入文件路径（特别提醒：Windows和Linux下路径的格式略有不同！分子文件中不要有空行！）然后敲击`Enter`后，程序会询问用户需要使用多少个进程（核）来并行计算，当用户输入想要调用的核数并敲击`Enter`后，程序会给出提示：**请键入q以优雅地退出程序，然后在终端输入下列命令**，用户需要根据所使用的系统选择输入哪条命令，以Windows为例，在终端输入`mpiexec -np 4 python .\gp_3x_mpirun.py -smiles_file_path ./gp_3x_test_mol/SMILES.txt -result_file_path mpi_batch_count_results.csv -task count`（**这里的命令只是例子，根据分子文件路径和想要调用的核数不同，屏幕上输出的命令也会有所不同，请根据实际情况随机应变**），会调用本程序中的`gp_3x_mpirun.py`模块进行并行计算，然后将结果写入生成在程序主目录的`mpi_batch_count_results.csv`中，其格式与2.2.1.3中导出的文件是一样的。



**注意：基于MPI并行批量计算分子的理化性质的速度优势仅在要计算的分子非常多时（如上万）才能体现，这是因为MPI多进程计算时进程之间相互通信会增加耗时，若要计算的分子数目不多（如几百上千），那么并行计算所减少的耗时无法抵消进程间相互通信所增加的耗时。更具体的例子，可以参考表2.1**



##### 2.2.1.8 与文件相关的操作(file)

基团贡献法虽然通用性强，计算速度快，但是其精度有限。因此，除了基团贡献法本身，本程序还向用户提供了一些用于可视化和生成分子动力学、量子化学计算（这在使用基团贡献法初筛，然后用高精度方法进一步筛选时非常有用）所需要的文件的功能。



###### 2.2.1.8.1 基于SMILES生成给定分子的xyz文件(file_1)

xyz文件（http://sobereva.com/477）几乎是最简单的记录分子三维结构的文件格式，几乎所有的可视化程序都可以打开它（如gaussview、MS、VESTA、VMD等）

若用户希望通过输入给定分子的SMILES从而得到对应分子的xyz文件，只需在主界面内输入`file`，然后在键盘上敲击`Enter`，进一步输入`1`在键盘上敲击`Enter`。然后根据后续的提示输入所需指令即可，如图2.10所示。

![](C:\Users\lrc\Desktop\group_contribution_3.2_dev\document\figure\main_file_1.png)

<div align = "center">图2.10 GC程序的主功能file_1</div>

用户在输入目标分子的SMILES后，程序会要求用户指定输出的xyz文件的路径，若用户直接敲`Enter`，则将会在程序的主目录下输出与输入SMILES同名的xyz文件（**注意：SMILES语法中的一些符号如：/，#，：不能出现在文件名，请自行修改**）。成功生成xyz文件后，用户可以输入`0`返回至主界面，然后输入`q`优雅地退出。



###### 2.2.1.8.2 批量生成xyz文件(file_2)

若用户需要生成一批分子的xyz文件，需要先准备一个2.1.2小节中介绍的分子文件，然后在主界面内输入`file`，然后在键盘上敲击`Enter`，然后输入`2`并敲击`Enter`，根据后续的提示输入所需指令即可，如图2.11所示。

![](C:\Users\lrc\Desktop\group_contribution_3.2_dev\document\figure\main_file_2.png)

<div align = "center">图2.11 GC程序的主功能file_2</div>

当用户进入主功能file的子功能2时，程序要求用户输入记录了一批分子的SMILES的文件，在用户输入文件路径（特别提醒：Windows和Linux下路径的格式略有不同！分子文件中不要有空行！）然后敲击`Enter`后，程序会自动读取分子文件中记录的SMILES。然后程序要求用户输入保存被生成出的xyz文件的根目录（即本次任务所有生成的xyz文件都保存在那里），用户输入之后开始生成xyz文件。在生成结束后，程序会在主目录下产生两个文件，分别是记录成功生成xyz文件的SMILES（xyz_succeed.txt）和没有成功生成xyz文件的SMILES（xyz_fail.txt）。



###### 2.2.1.8.3 基于MPI并行批量生成xyz文件(file_-2)

现代计算机的中央处理器（CPU）往往是多核的，若在批量生成xyz文件时希望充分利用CPU的性能，可以使用MPI并计算。

若用户需要生成一批分子的xyz文件，需要先准备一个2.1.2小节中介绍的分子文件，然后在主界面内输入`file`，然后在键盘上敲击`Enter`，在输入`-2`并敲击`Enter`，根据后续的提示输入所需指令即可，如图2.12所示。

![](C:\Users\lrc\Desktop\group_contribution_3.2_dev\document\figure\main_file_mins2_01.png)

<div align = "center">图2.12 GC程序的主功能file_-2 (生成命令阶段)</div>

程序会提示用户键入保存了分子SMILES的文件路径。用户输入文件路径（特别提醒：Windows和Linux下路径的格式略有不同！分子文件中不要有空行！）然后敲击`Enter`。之后程序要求用户输入保存被生成出的xyz文件的根目录（即本次任务所有生成的xyz文件都保存在那里），用户输入之后开始生成xyz文件。接着程序会询问用户需要使用多少个进程（核）来并行计算，当用户输入想要调用的核数并敲击`Enter`后，程序会给出提示：**请键入q以优雅地退出程序，然后在终端输入下列命令**（如图2.13所示），用户需要根据所使用的系统选择输入哪条命令，以Windows为例，在终端输入`mpiexec -np 4 python .\gp_3x_mpirun.py -smiles_file_path ./gp_3x_test_mol/SMILES.txt -out_root_path mpi_test_xyz -task xyz`（**这里的命令只是例子，根据分子文件路径和想要调用的核数不同，屏幕上输出的命令也会有所不同，请根据实际情况随机应变**），会调用本程序中的`gp_3x_mpirun.py`模块进行并行计算，然后将所有生成xyz文件生成在程序主目录的`mpi_test_xyz`目录中。在生成结束后，程序会在主目录下产生两个文件，分别是记录成功生成xyz文件的SMILES（xyz_succeed.txt）和没有成功生成xyz文件的SMILES（xyz_fail.txt）。

![](C:\Users\lrc\Desktop\group_contribution_3.2_dev\document\figure\main_file_mins2_02.png)

<div align = "center">图2.13 GC程序的主功能file_-2 (运行命令阶段)</div>

先输入`0`返回至主界面，然后再输入`q`优雅地退出程序。在终端输入刚才程序给我们生成的命令即可。



###### 2.2.1.8.4 转化给定文件的文件格式(file_3)

本程序还提供了转换文件格式的功能，用户只需提供要被转化为原文件格式和路径，然后指定需要的文件格式和路径即可。**本程序此功能基于openbabel，因此但凡openbabel支持的文件格式，本功能都支持，如：xyz，mol，mol2，pdb...**

使用例子见图2.14。

![](C:\Users\lrc\Desktop\group_contribution_3.2_dev\document\figure\main_file_3.png)

<div align = "center">图2.14 GC程序的主功能file_3 </div>

注意，当指定目标文件路径时，若直接敲`Enter`，则会在程序主目录下生成与原文件同名的文件。



###### 2.2.1.8.5 批量转化给定文件的文件格式(file_4)

当需要转化一批文件的文件格式时，首先应把待转换格式的文件放入同一个文件夹。然后进入主功能`file`的子功能`4`，首先程序会询问用户待转换的文件格式如何，然后还需输入待转换格式的文件的根目录（即保存了所有带转换格式的文件的目录）。接着程序问用户希望将文件转化为什么格式，用户输入之后（如xyz，mol，mol2，pdb，gro等），程序要求用户指定转化之后的新文件的根目录（即所有新产生的文件都将保存在那里），若直接敲击`Enter`，将会使用待转换格式的文件的根目录。具体操作见图2.15。

![](C:\Users\lrc\Desktop\group_contribution_3.2_dev\document\figure\main_file_4.png)

<div align = "center">图2.15 GC程序的主功能file_4 </div>



###### 2.2.1.8.6 基于SMILES生成gjf文件(file_5)

本程序还为用户提供了基于分子的SMILES生成对应的gjf文件（量子化学计算软件Gaussian的输入文件）。用户只需进入主功能file中的子功能5，然后根据提示操作即可，见图2.16。

![](C:\Users\lrc\Desktop\group_contribution_3.2_dev\document\figure\main_file_5.png)

<div align = "center">图2.16 GC程序的主功能file_5 </div>

若用户对这里提供的接口（如高斯调用的CPU核数、内存数目、关键词、chk文件路径）不甚了解，可以参考量子化学软件Gaussian的用户手册或相关论文，也可联系开发者，或可得到一些帮助。



###### 2.2.1.8.7 批量生成gjf文件(file_6)

还可以批量生成分子的gjf文件，程序要求用户输入记录了一批分子的SMILES的文件，在用户输入文件路径（特别提醒：Windows和Linux下路径的格式略有不同！分子文件中不要有空行！）然后敲击`Enter`后，程序会自动读取分子文件中记录的SMILES。然后程序要求用户输入保存被生成出的gjf文件的根目录（即本次任务所有生成的gjf文件都保存在那里），用户输入之后开始生成gjf文件。在生成结束后，程序会在主目录下产生两个文件，分别是记录成功生成gjf文件的SMILES（gjf_succeed.txt）和没有成功生成gjf文件的SMILES（gjf_fail.txt）。具体操作步骤见图2.17。

![](C:\Users\lrc\Desktop\group_contribution_3.2_dev\document\figure\main_file_6.png)

<div align = "center">图2.17 GC程序的主功能file_6 </div>



###### 2.2.1.8.8 基于MPI并行批量生成生成gjf文件(file_-6)

现代计算机的中央处理器（CPU）往往是多核的，若在批量生成gjf文件时希望充分利用CPU的性能，可以使用MPI并计算。

若用户需要生成一批分子的gjf文件，需要先准备一个2.1.2小节中介绍的分子文件，然后在主界面内输入`file`，然后在键盘上敲击`Enter`，在输入`-6`并敲击`Enter`，根据后续的提示输入所需指令即可，如图2.18所示。

![](C:\Users\lrc\Desktop\group_contribution_3.2_dev\document\figure\main_file_mins6_01.png)

<div align = "center">图2.18 GC程序的主功能file_-6 （生成命令阶段）</div>



![](C:\Users\lrc\Desktop\group_contribution_3.2_dev\document\figure\main_file_mins6_02.png)

<div align = "center">图2.19 GC程序的主功能file_-6 （执行命令阶段）</div>

先输入`0`返回至主界面，然后再输入`q`优雅地退出程序。在终端输入刚才程序给我们生成的命令即可。

 

#### 2.2.2 GC作为外部库

本程序还可作为外部库导入进用户自行编写的python脚本中，下面分别介绍。

##### 2.2.2.1 计算单个分子的理化性质

只需按如下方式编写python脚本即可：

```python
from gp_3x_calculator import Calculator  # 从Group Contribution导入计算器


calculator = Calculator()  # 实例化一个Calculator对象
print(calculator.calculate_a_mol('CC1(C2)CC(C3)CC2CC3(C)C1', debug=True))  # 打印计算结果，这里设置debug=True是要求程序同时打印基团数目
```



##### 2.2.2.2 统计单个分子的基团数目

只需按如下方式编写python脚本即可：

```python
from gp_3x_counter import Counter  # 从Group Contribution导入统计器


counter = Counter()  # 实例化一个Counter对象
result = c.count_a_mol(m, clear_mode=True)  # 统计结果，使用清爽模式
print(result)  # 打印计算结果
```



##### 2.2.2.3 批量计算分子

我们这里以Group Contribution内部数据文件夹中的`gdb.txt`(记录了大约三十万个饱和碳氢分子的SMILES)为例，对其进行批量计算。假定在当前工作目录输出名为`test.csv`的结果文件。

```python
import os
from gp_3x_calculator import Calculator  # 从Group Contribution导入计算器


calculator = Calculator()  # 实例化一个Calculator对象
input_file_path = os.path.join('gp_3x_internal_data', 'gdb.txt')  # 说明输入文件路径
output_file_path = os.path.join('test.csv')  # 说明输出文件路径
calculator.calculate_mols(input_file_path, output_file_path)  # 开始计算
```



##### 2.2.2.4 基于MPI并行批量计算分子

除了单核运行模型外，Group Contribution还支持多核并行批量计算分子性质。我们给出名为`gp_3x_mpirun.py`的运行脚本

以Group Contribution内部数据文件夹中的`gdb.txt`(记录了大约三十万个饱和碳氢分子的SMILES)为例，对其进行批量计算，且在当前工作目录输出名为`mpi_batch_calculate_results.csv`的结果文件。

**必须**在终端输入以下命令（强烈建议在Linux系统中运行）：

```sh
# 使用8个进程并行计算
# 注意Windows和Linux路径的区别

```

程序将会在当前目录输出名为`mpi_batch_calculate_results.csv`的结果文件。



##### 2.2.2.5 批量统计分子基团数目

我们这里以Group Contribution内部数据文件夹中的`SMILES.txt`(记录了几百个各类分子的SMILES)为例，对其进行批量计算。假定在当前工作目录输出名为`count_result.csv`的结果文件。

```python
import os 
from gp_3x_counter import Counter  # 从Group Contribution导入统计器


c = Counter()  # 实例化一个Counter对象
c.count_mols(smiles_file_path=os.path.join('gp_3x_test_mol', 'SMILES.txt'), 
             count_result_file_path='count_result.csv', 
             add_note=True,
             add_smiles=True
            )  # 批量统计分子基团数目
```



##### 2.2.2.6 基于MPI并行批量统计分子基团数目

除了单核运行模型外，Group Contribution还支持多核并行批量计算分子性质。我们给出名为`gp_3x_mpirun.py`的运行脚本

以Group Contribution内部数据文件夹中的`gdb.txt`(记录了大约三十万个饱和碳氢分子的SMILES)为例，对其进行批量计算，且在当前工作目录输出名为`result_mpi.csv`的结果文件。

**必须**在终端输入以下命令（强烈建议在Linux系统中运行）：

```sh
# 使用8个进程并行计算
# 注意Windows和Linux路径的区别

```

程序将会在当前目录输出名为`mpi_batch_count_results.csv`的结果文件。



##### 2.2.2.7 与文件相关的操作

###### 2.2.2.7.1 基于SMILES生成给定分子的xyz文件

```python
from gp_3x_tool import Tool  # 从Group Contribution中导入工具箱Tool


t = Tool()  # 实例化一个Tool对象
t.smi_to_xyz('C1CCCC1', xyz_path='test.xyz')  # 生成C1CCCC1的xyz文件，路径为test.xyz
```



###### 2.2.2.7.2 批量生成xyz文件

```python
from gp_3x_tool import Tool  # 从Group Contribution中导入工具箱Tool


t = Tool()  # 实例化一个Tool对象
t.batch_smi_to_xyz(smiles_file_path='SMILES.txt', xyz_root_path='test_xyz')  # 批量生成xyz文件，生成的xyz都保存在test_xyz中
```



###### 2.2.2.7.3 基于MPI并行批量生成xyz文件

除了单核运行模型外，Group Contribution还支持多核并行批量生成xyz文件。我们给出名为`gp_3x_mpirun.py`的运行脚本

以Group Contribution内部数据文件夹中的`SMILES.txt`(记录了几百个各类分子的SMILES)为例，批量计算xyz文件，且在当前工作目录中的mpi_test_xyz目录下输出所有xyz文件。

**必须**在终端输入以下命令（强烈建议在Linux系统中运行）：

```sh
# 使用8个进程并行计算
# 注意Windows和Linux路径的区别

```



###### 2.2.2.7.4 转化给定文件的文件格式

```python
from gp_3x_tool import Tool  # 从Group Contribution中导入工具箱Tool


t = Tool()  # 实例化一个Tool对象
t.convert_file_type(in_format='xyz', in_path='in.xyz', 
                    out_format='mol2', out_path='out.mol2')
```



###### 2.2.2.7.5 批量转化给定文件的文件格式

```python
from gp_3x_tool import Tool  # 从Group Contribution中导入工具箱Tool


t = Tool()  # 实例化一个Tool对象
t.batch_convert_file_type(in_format='xyz', in_root_path='in_xyz', 
                          out_format='mol2', out_root_path='out_mol2')
```



###### 2.2.2.7.6 基于SMILES生成gjf文件

```python
from gp_3x_tool import Tool  # 从Group Contribution中导入工具箱Tool


t = Tool()  # 实例化一个Tool对象
t.smi_to_gjf(smi='C1CCC1', nproc='12', mem='12GB', 
             gaussian_keywords='#p opt freq b3lyp/6-31g*', 
             chk_path='C1CCC1.chk', 
             gjf_path='C1CCC1.gjf', 
             add_other_std_tasks=False)
```



###### 2.2.2.7.7 批量生成gjf文件

```python
from gp_3x_tool import Tool  # 从Group Contribution中导入工具箱Tool


t = Tool()  # 实例化一个Tool对象
t.batch_smi_to_gjf(smiles_file_path='SMILES.txt', gjf_root_path='test_gjf',
                   nproc='12', mem='12GB', 
                   gaussian_keywords='#p opt freq b3lyp/6-31g*', 
                   add_other_std_tasks=False)
```



###### 2.2.2.7.8 基于MPI并行批量生成生成gjf文件

除了单核运行模型外，Group Contribution还支持多核并行批量生成gjf文件。我们给出名为`gp_3x_mpirun.py`的运行脚本

以Group Contribution内部数据文件夹中的`SMILES.txt`(记录了几百个各类分子的SMILES)为例，批量计算gjf文件，且在当前工作目录中的mpi_test_gjf目录下输出所有gjf文件。

**必须**在终端输入以下命令（强烈建议在Linux系统中运行）：

```sh
# 使用8个进程并行计算
# 注意Windows和Linux路径的区别

```



## 3 高级

### 3.1 导出基团贡献法风格的分子指纹

除了计算分子性质，Group Contribution还支持直接导出分子不同基团数目的统计结果，即输出基团贡献法风格的分子指纹。

```python
from gp_3x_counter import Counter  # 从Group Contribution中导入计数器Counter


c = Counter()  # 实例化一个Counter对象
result = c.count_a_mol(m, clear_mode=False)  # 输出基团数目的统计结果。clear_mode=False是指不省略数目为0的基团
print(result)

print(c.get_group_fingerprint(m))  # 直接输出列表形式的统计结果，如：[1,2,1,0,0,5.....]。该列表的长度为所有基团的种类数（220 + 130 + 74）
```



#### 3.1.1 基于基团贡献法风格的分子指纹建立机器学习模型（todo）



### 3.2 重新拟合基团参数（todo）



