# Data-Science-Thesis-codes
This repository contains all the code used for my thesis project on forecasting electricity consumption at the regional level in France. In this work, i compare three models (LSTM, GRU, and XGBoost) across four prediction horizons: 1-hour, 8-hour, 1-day, and 7-day ahead. The experiments are conducted on two datasets:
- A basic dataset without exogenous features
- An enriched dataset with exogenous features


# Datasets used
The original dataset used at the start of this thesis was too large to upload to GitHub and is therefore not included in this repository. It can be accessed here: https://www.kaggle.com/datasets/ravvvvvvvvvvvv/france-energy-weather-hourly

The dataset used for training and testing the models are available in this repository, along with the EDA dataset. They are:
- Hourly_Elec_Original_Data.pkl ( basic dataset)
- Hourly_Elec_Additional_Data.pkl (enriched dataset)
These two datasets can be used to run LSTM, GRU, and XGBoost models for performance evaluation.
Additionally, the EDA dataset is available for analysis:
- Hourly_Elec_Original_EDA.pkl 

# Repository structure
The repository contains the following main components:
**Data Cleaning**
A Jupyter notebook to clean the original dataset from the Kaggle source.
**Dataset Merging**
A folder containing all CSV files needed to add exogenous features.
A corresponding Jupyter notebook to merge these datasets.
**Feature Engineering**
A Jupyter notebook to process the data and remove unnecessary features.

**Model Files**
XGBoost_files: Jupyter notebooks that can be run in JupyterLab or Visual Studio Code.
LSTM_files and GRU_files: Python scripts that can be run in Visual Studio Code by executing the main function in each script.
