from setuptools import setup, find_packages
from os.path import join


VERSION = '3.0.0'
DESCRIPTION = 'Groupy -- A Useful Tool for Molecular Analysis'

setup(
    name="Groupy",
    version=VERSION,
    author="Ruichen Liu",
    author_email="1197748182@qq.com",
    description=DESCRIPTION,
    url='https://github.com/47-5/Groupy',

    packages=find_packages(),

    install_requires=['rdkit', 'pandas', 'numpy', 'tqdm', 'ase', 'joblib', 'openpyxl'],
    python_requires='>3.5',

    data_files=[(join('.', 'groupy'), [join('.', 'groupy', 'group_contribution_parameters.xlsx'),
                                                           join('.', 'groupy', 'group_order.xlsx')
                                                           ])],


    entry_points={
            'console_scripts': [
                'Groupy = groupy.groupy:main'
            ]
        },
    scripts=['groupy/groupy.py'],

)