#!/bin/bash

# Activate the conda environment
source activate 215a

# Update the conda environment with environment.yaml
conda env update --file environment.yaml --prune

# Run Data Cleaning
python /clean.py

# Run EDA
python notebook /EDA.ipynb

# Run Data Perturbation
python notebook /perturbed.ipynb

# Run Models with Cleaned Data
jupyter notebook /cnn.ipynb

jupyter notebook /logreg.ipynb

jupyter notebook /neural_network.ipynb

jupyter notebook /qda.ipynb

jupyter notebook /qda_class_balance.ipynb

jupyter notebook /randomforest_class_balance.ipynb

# Run Models with Perturbed Data

jupyter notebook /cnn_perturbed.ipynb
