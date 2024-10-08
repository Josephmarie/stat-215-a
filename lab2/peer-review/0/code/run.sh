#!/bin/bash
conda activate 215a
jupyter nbconvert --to notebook --execute --inplace Data_Clean.ipynb
jupyter nbconvert --to notebook --execute --inplace EDA.ipynb
jupyter nbconvert --to notebook --execute --inplace PCA.ipynb
jupyter nbconvert --to notebook --execute --inplace Clustering1.ipynb
jupyter nbconvert --to notebook --execute --inplace Clustering2.ipynb
