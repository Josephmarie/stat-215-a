#!/bin/bash
conda activate 215a
# executes a Jupyter notebook containing some analysis
jupyter nbconvert --to notebook --execute --inplace lab2.ipynb
