#!/bin/bash

# run.sh

# --
# Setup environment

conda create -y -n uclasm_env python=3.7
conda activate uclasm_env

# ?? versions
pip install numpy==1.21.1
pip install scipy==1.7.0
pip install pandas==1.3.0
pip install tqdm==4.61.2
pip install matplotlib==3.4.2
pip install networkx==2.5.1
pip install setuptools

pip install -e .