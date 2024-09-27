from setuptools import setup, find_packages
import os


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

    data_files=[]
)