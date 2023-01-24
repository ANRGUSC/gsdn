#!/bin/bash

sudo apt update && sudo apt install -y libgraphviz-dev
conda init
conda env create --file ./environment.yml
echo "conda activate gsdn" >> /home/gitpod/.bashrc