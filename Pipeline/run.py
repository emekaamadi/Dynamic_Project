#Run this file to run the entire pipeline

import pandas as pd
import numpy as np
import subprocess
import seaborn as sns
import matplotlib.pyplot as plt
from preprocess import *
from train import *

print("Completed Initialization!")
# Load in Data Sets and Merge/Clean
data_df = load_and_preprocess_cab_data("../cab_rides.csv")
weather_df = load_and_preprocess_weather_data("../weather.csv")
print("Loading and Cleaning the Data!")

if data_df is not None and weather_df is not None:
    merged_df = merge_and_clean_data(data_df, weather_df)
    # Save the cleaned data
    merged_df.to_csv("Data/base_cleaned.csv")
print("Started Training Base Model!")

# Process and save model for base data
base_data = get_base_data(merged_df)
X, y, preprocessor = prepare_data(base_data)
train_and_save_model(X, y, preprocessor, 'base_model')
print("Saved Base Model")

# Process and save model for dynamic data
dynamic_data = get_dynamic_data(merged_df)
X, y, preprocessor = prepare_data(dynamic_data)
train_and_save_model(X, y, preprocessor, 'dynamic_model')
print("Saved Dynamic Model")

# Creation of Demand Data via Demand Estimation takes a few hours to run only uncomment the following lines to get new demand estimation that is currently stored in Data Folder.
#eta_df = save_demand_data()   

# Process and save model for demand data 
demand_data = get_demand_data()
X, y, preprocessor = prepare_data(demand_data)
train_and_save_model(X, y, preprocessor, 'demand_model')
print("Saved Demand Model")  

# Process and save model for eta estimation
demand_data_eta = get_demand_data_with_eta()
X, y, preprocessor = prepare_data_eta(demand_data_eta)
train_and_save_model_for_eta(X, y, preprocessor, 'demand_model_eta')
print("Saved eta-estimation model")

# Running the Streamlit app
subprocess.run(["streamlit", "run", "app.py"])
