#Run this file to run the entire pipeline

# Check if Data/demand_est.csv exists
# If not, run demand_estimation.py to create it
import os
import subprocess
if not os.path.exists("Data/demand_est.csv"):
    print("Data/demand_est.csv does not exist, generating it now")
    subprocess.run(["python", "demand_estimation.py"])
    eta_df = pd.read_csv("Data/demand_est.csv")
else:
    print("Data/demand_est.csv already exists")


# Import packages
import pandas as pd
import numpy as np
import subprocess
import seaborn as sns
import matplotlib.pyplot as plt
from preprocess import *
from train import *
from demand_estimation import *


print("Completed Initialization!")
# Load in Data Sets and Merge/Clean
data_df = load_and_preprocess_cab_data("Data/cab_rides.csv")
weather_df = load_and_preprocess_weather_data("Data/weather.csv")
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

# Process and save model for demand data 
demand_data = get_demand_data(data=eta_df)
X, y, preprocessor = prepare_data(demand_data)
train_and_save_model(X, y, preprocessor, 'demand_model')
print("Saved Demand Model")  

# Process and save model for eta estimation
demand_data_eta = get_demand_data_with_eta(data=pd.read_csv("demand_est.csv"))
X, y, preprocessor = prepare_data_eta(demand_data_eta)
train_and_save_model_for_eta(X, y, preprocessor, 'demand_model_eta')
print("Saved eta-estimation model")

# Running the Streamlit app
subprocess.run(["streamlit", "run", "app.py"])
