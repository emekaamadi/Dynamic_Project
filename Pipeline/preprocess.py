import pandas as pd
import numpy as np

# Function to load and preprocess cab ride data
def load_and_preprocess_cab_data(filepath):
    """
    Load cab ride data from a CSV file and preprocess it.
    Args:
        filepath (str): The path to the cab ride data CSV file.
    Returns:
        DataFrame: Preprocessed cab ride data.
    """
    try:
        data_df = pd.read_csv(filepath)
        data_df = data_df[data_df['name'] != 'Taxi']
        data_df['date_time'] = pd.to_datetime(data_df['time_stamp'], unit='ms')
        data_df['car_type'] = data_df['name'].apply(determine_car)
        data_df['weekday'] = data_df['date_time'].dt.weekday.apply(lambda x: 1 if 0 <= x <= 4 else 0)
        data_df['rush_hour'] = data_df['date_time'].apply(is_rush_hour)
        return data_df
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None

# Function to determine car type
def determine_car(car):
    car_mapping = {
        "Black": "Luxury", "Lux Black": "Luxury", "Lux": "Luxury",
        "Black SUV": "Luxury SUV", "Lux Black XL": "Luxury SUV",
        "UberX": "Base", "Lyft": "Base",
        "UberXL": "Base XL", "Lyft XL": "Base XL",
        "UberPool": "Shared", "Shared": "Shared",
        "WAV": "Wheel Chair Accessible"
    }
    return car_mapping.get(car, "Other")

# Function to check if it's rush hour
def is_rush_hour(time_obj):
    morning_rush = time_obj.hour in range(7, 10)
    evening_rush = time_obj.hour in range(16, 19)
    return int(morning_rush or evening_rush)

# Function to load and preprocess weather data
def load_and_preprocess_weather_data(filepath):
    """
    Load weather data from a CSV file and preprocess it.
    Args:
        filepath (str): The path to the weather data CSV file.
    Returns:
        DataFrame: Preprocessed weather data.
    """
    try:
        weather_df = pd.read_csv(filepath)
        weather_df['date_time'] = pd.to_datetime(weather_df['time_stamp'], unit='s')
        weather_df['is_raining'] = weather_df['rain'].apply(lambda x: 1 if x > 0 else 0)
        weather_df['temp_groups'] = weather_df['temp'].apply(group_temp)
        return weather_df
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None

# Function to group temperature
def group_temp(temp):
    temp_ranges = [(30, 20), (40, 30), (50, 40), (float('inf'), 50)]
    for upper_bound, group in temp_ranges:
        if temp < upper_bound:
            return group

# Function to merge and clean datasets
def merge_and_clean_data(ride_data, weather_data):
    """
    Merge and clean ride and weather datasets.
    Args:
        ride_data (DataFrame): Preprocessed ride data.
        weather_data (DataFrame): Preprocessed weather data.
    Returns:
        DataFrame: Merged and cleaned dataset.
    """
    merged_df = pd.merge_asof(ride_data.sort_values('date_time'), 
                              weather_data.sort_values('date_time'), 
                              on='date_time', 
                              left_by='source', 
                              right_by='location',
                              direction='nearest')

    final_columns = [
        'date_time', 'distance', 'cab_type', 'source', 'destination',
        'car_type', 'weekday', 'rush_hour', 'is_raining', 'temp_groups',
        'surge_multiplier', 'price',
    ]

    return merged_df[final_columns]

# Function to get main data 
def get_cleaned_data():
    """
    Loads, merges, and cleans the main datasets.
    Returns:
        DataFrame: The cleaned data.
    """
    data_df = load_and_preprocess_cab_data("../cab_rides.csv")
    weather_df = load_and_preprocess_weather_data("../weather.csv")
    if data_df is not None and weather_df is not None:
        return merge_and_clean_data(data_df, weather_df)
    else:
        raise ValueError("Error in loading data.")
    
# Function to get base modeling data
def get_base_data(data=get_cleaned_data()):
    """
    Extract base data where surge_multiplier is 1.0.
    Returns:
        DataFrame: Filtered base data.
    """
    df = data
    base_df = df[df["surge_multiplier"] == 1.0]
    return base_df[["cab_type", "source", "destination", "car_type", "weekday", "rush_hour", "is_raining", "temp_groups", "price"]]

# Function to get dynamic modeling data 
def get_dynamic_data(data=get_cleaned_data()):
    """
    Extract dynamic data.
    Returns:
        DataFrame: Filtered dynamic data.
    """
    df = data
    return df[["cab_type", "source", "destination", "car_type", "weekday", "rush_hour", "is_raining", "temp_groups", "price"]]
 
def get_demand_data():
    """
    Create dataset based off demand estimation calculation.
    Returns:
        DataFrame: Filtered Demand data.
    """
    # Load in already saved data because it takes hours to do the demand estimation
    df = pd.read_csv("Data/demand_est.csv")
    df['estimated_demand'] = df['estimated_a'] * np.exp(-np.abs(df['estimated_eta']) * np.log(df['price'])) + df['estimated_b']
    df['base_price'] = df.groupby(by=['source', 'destination', 'car_type'])['price'].transform('min')
    df.rename(columns = {'price': 'original_price'}, inplace= True)
    # Now the new price column will actually be the dynamic price
    df['price'] = df['base_price'] * (1 + df['estimated_eta'] * df['estimated_demand'])
    return df[["cab_type", "source", "destination", "car_type", "weekday", "rush_hour", "is_raining", "temp_groups", "price"]]

# # Main script execution
# if __name__ == "__main__":
#     data_df = load_and_preprocess_cab_data("../cab_rides.csv")
#     weather_df = load_and_preprocess_weather_data("../weather.csv")

#     if data_df is not None and weather_df is not None:
#         merged_df = merge_and_clean_data(data_df, weather_df)
#         # Save the cleaned data
#         merged_df.to_csv("Data/base_cleaned.csv")
