import pandas as pd
import numpy as np
import os


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
    data_df = load_and_preprocess_cab_data("Data/cab_rides.csv")
    weather_df = load_and_preprocess_weather_data("Data/weather.csv")
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


def get_demand_data(eta_df="Data/demand_est.csv"):
    """
    Create dataset based off demand estimation calculation.
    Returns:
        DataFrame: Filtered Demand data.
    """
    # Load in already saved data because it takes hours to do the demand estimation
    # If Data/demand_est.csv does not exist, then make df = save_demand_data()
    # if os.path.isfile("Data/demand_est.csv"):
    #     print("Demand Estimation Data Found!")
    # else:
    #     print("No Demand Estimation Data Found!, Generating New Data!")
    #     eta_df = save_demand_data()
    #     print("Saved Demand Estimation Data!")
    df = eta_df

    df['estimated_demand'] = df['estimated_a'] * np.exp(-np.abs(df['estimated_eta']) * np.log(df['price'])) + df['estimated_b']
    df['base_price'] = df.groupby(by=['source', 'destination', 'car_type'])['price'].transform('min')
    df.rename(columns = {'price': 'original_price'}, inplace= True)
    # Now the new price column will actually be the dynamic price
    df['price'] = df['base_price'] * (1 + df['estimated_eta'] * df['estimated_demand'])
    if "date_time" in df.columns:
        # Set date_time as index
        df.set_index('date_time', inplace=True)
        keep = ["cab_type", "source", "destination", "car_type", "weekday", "rush_hour", "is_raining", "temp_groups", "price"]
    else:
        keep = ["cab_type", "source", "destination", "car_type", "weekday", "rush_hour", "is_raining", "temp_groups", "price"]
    return df[keep]


def get_demand_data_with_eta(eta_df="Data/demand_est.csv"):
    """
    Create dataset based off demand estimation calculation.
    Returns:
        DataFrame: Filtered Demand data.
    """
    # Load in already saved data because it takes hours to do the demand estimation
    # if os.path.isfile("Data/demand_est.csv"):
    #     print("Demand Estimation Data Found!")
    # else:
    #     print("No Demand Estimation Data Found!, Generating New Data!")
    #     eta_df = save_demand_data()
    #     print("Saved Demand Estimation Data!")
    df = eta_df

    df['estimated_demand'] = df['estimated_a'] * np.exp(-np.abs(df['estimated_eta']) * np.log(df['price'])) + df['estimated_b']
    df['base_price'] = df.groupby(by=['source', 'destination', 'car_type'])['price'].transform('min')
    df.rename(columns = {'price': 'original_price'}, inplace= True)
    # Now the new price column will actually be the dynamic price
    df['price'] = df['base_price'] * (1 + df['estimated_eta'] * df['estimated_demand'])
    if "date_time" in df.columns:
        # Set date_time as index
        df.set_index('date_time', inplace=True)
        keep = ["cab_type", "source", "destination", "car_type", "weekday", "rush_hour", "is_raining", "temp_groups", "price", "estimated_eta"]
    else:
        keep = ["cab_type", "source", "destination", "car_type", "weekday", "rush_hour", "is_raining", "temp_groups", "price", "estimated_eta"]
    return df[keep]


def get_MCMC_data(whole_data="Data/demand_est.csv"):
    # if os.path.isfile("Data/demand_est.csv"):
    #     print("Demand Estimation Data Found!")
    # else:
    #     print("No Demand Estimation Data Found!, Generating New Data!")
    #     eta_df = save_demand_data()
    #     print("Saved Demand Estimation Data!")
    df = whole_data
    return whole_data[["cab_type", "source", "destination", "car_type", "weekday", "rush_hour", "is_raining", "temp_groups", "price", "estimated_eta", "estimated_a", "estimated_b"]]


def get_estimated_values(MCMC_data, input_df=get_dynamic_data()):
    MCMC_data = get_MCMC_data()
    filters = pd.Series([True] * len(MCMC_data))  
    for col in input_df.columns:
        filters &= (MCMC_data[col] == input_df.at[0, col]) 
    filtered = MCMC_data[filters]
    if filtered.empty:
        return "There's no data matched", 0, 0
    else:
        return filtered.iloc[0]['estimated_eta'], filtered.iloc[0]['estimated_a'], filtered.iloc[0]['estimated_b']


def col_item_dict(data=get_dynamic_data()):
    df = data
    unique_dict = {col: df[col].unique().tolist() for col in df.columns}
    return unique_dict


def get_service_types():
    car_type_options = {
        'Uber': ['Luxury', 'Base XL', 'Base', 'Wheel Chair Accessible', 'Luxury SUV', 'Shared'],
        'Lyft': ['Luxury SUV', 'Base', 'Shared', 'Luxury', 'Base XL']
    }
    return car_type_options


def get_questions_answers():
    unique_values = col_item_dict()  
    questions = ['Uber or Lyft?', 'Where are customers coming from?', 'Where are customers going?', 'What type of service?', 'Weekday or Weekend?', 'Is it rush hour?', 'Is it raining?', 'What is the temperature group?']
    answers = [unique_values['cab_type'], unique_values['source'], unique_values['destination'], unique_values['car_type'], ['Weekday', 'Weekend'], ['Yes', 'No'], ['Yes', 'No'], ['20-30 degrees', '30-40 degrees', '40-50 degrees', '50 or more']]
    return questions, answers


def option_translator(option_list):
    temp_dict = {'20-30 degrees': 20, '30-40 degrees': 30, '40-50 degrees': 40, '50 or more': 50}
    week_dict = {'Weekday': 1, 'Weekend': 0}
    yn_dict = {'Yes': 1, 'No': 0}
    cols = ['cab_type', 'source', 'destination', 'car_type', 'weekday', 'rush_hour', 'is_raining', 'temp_groups']
    df = pd.DataFrame(columns=cols)
    df.loc[0] = option_list
    df['temp_groups'] = df['temp_groups'].map(temp_dict)
    df['weekday'] = df['weekday'].map(week_dict)
    df['rush_hour'] = df['rush_hour'].map(yn_dict)
    df['is_raining'] = df['is_raining'].map(yn_dict)
    return df


def adjust_demand_price(base_price, dynamic_price, demand_price):
    # find the max and min of the other two prices
    other_prices = [base_price, dynamic_price]
    max_other_price = max(other_prices)
    min_other_price = min(other_prices)

    # if demand price is the max of the three prices
    if demand_price == max(base_price, dynamic_price, demand_price):
        # adjust demand price to be 20% less than the max of the other two prices
        if demand_price > 1.2 * min_other_price:
            demand_price = 1.2 * min_other_price

    # if demand price is the min of the three prices
    elif demand_price == min(base_price, dynamic_price, demand_price):
        # adjust demand price to be 20% more than the min of the other two prices
        if demand_price < 0.8 * max_other_price:
            demand_price = 0.8 * max_other_price

    return base_price, dynamic_price, demand_price
    

# ### To remove co-dependency on demand_estimation.py, we have to copy the following functions from demand_estimation.py ###
import pymc3 as pm
import theano.tensor as tt
import pandas as pd
import numpy as np


def load_MCMC_df(data = get_cleaned_data()):
    if "Unnamed: 0" in data.columns: 
        data.drop(columns=["Unnamed: 0"], axis=1, inplace=True)
    if "date_time" in data.columns:
        # Set date_time as index
        data.set_index('date_time', inplace=True)
        # Sort by index
        data.sort_index(inplace=True) 
    return data


def estimate_demand_parameters(dataframe, price_col):
    """
    ### Estimates the demand parameters (eta, a, b) using PyMC3 and adds them as new columns in the dataframe.
    :param dataframe: Pandas DataFrame containing the data
    :param price_col: Name of the column containing price data
    :return: DataFrame with new columns for eta, a, b estimates
    """
    with pm.Model() as model:
        # Define priors
        eta_shape = 2
        eta_rate = 5

        ### Set the prior for eta to be a Gamma distribution with shape=2 and rate=5
        eta = pm.Gamma('eta', alpha=eta_shape, beta=eta_rate) + 0.25

        ### Set the priors for a and b
        a = pm.Uniform('a', lower=0, upper=100)
        b = pm.Uniform('b', lower=0, upper=20)

        # Convert the price data to a theano tensor
        price_data = tt.as_tensor_variable(dataframe[price_col].values)
        surge_multiplier_data = tt.as_tensor_variable(dataframe['surge_multiplier'].values)

        # Define the demand function using pm.math.exp for exponentiation
        demand = a * pm.math.exp(-tt.abs_(eta) * tt.log(price_data)) + b

        # Assuming Gaussian noise in the observed data
        observed = pm.Normal('observed', mu=demand, sd=1, observed=dataframe[price_col])

        # Sample from the posterior
        trace = pm.sample(1000, tune=1000, return_inferencedata=False)

    # Extracting the mean of the posterior distributions
    eta_mean = np.mean(trace['eta'])
    a_mean = np.mean(trace['a'])
    b_mean = np.mean(trace['b'])

    # Adding new columns to the dataframe
    dataframe['estimated_eta'] = eta_mean
    dataframe['estimated_a'] = a_mean
    dataframe['estimated_b'] = b_mean

    return dataframe, trace


def save_demand_data(data=load_MCMC_df()):
    lst_car_type = list(set(data.car_type))
    lst_source = list(set(data.source))
    lst_destination = list(set(data.destination))
    results_list = []

    # Iterate through all combinations of car_type, source, and destination
    for car_type in lst_car_type:
        for source in lst_source:
            for destination in lst_destination:
                # Filter the data for the current combination
                filtered_data = data[(data['car_type'] == car_type) & (data['source'] == source) & (data['destination'] == destination)]

                # If the filtered data is not empty, estimate the demand parameters
                if not filtered_data.empty:
                    try:
                        # Estimate the demand parameters and add them to the dataframe
                        result_df, _ = estimate_demand_parameters(filtered_data, 'price')
                        results_list.append(result_df)
                    # except Exception as e:
                    #     print(f"Error processing combination: CarType={car_type}, Source={source}, Destination={destination}")
                    #     print(f"Error message: {e}")
                    except Exception as e:
                        print(f"Error processing combination: CarType={car_type}, Source={source}, Destination={destination}")
                        print(f"Error message: {e}")

                        # Generate default values for the demand parameters
                        default_values = {'estimated_eta': 0.4, 'estimated_a': 0.5, 'estimated_b': 30}
                        default_df = pd.DataFrame(default_values, index=[0])
                        default_df = pd.concat([filtered_data.iloc[:1].drop(['price'], axis=1), default_df], axis=1)
                        results_list.append(default_df)


    # Combine all the results into a single dataframe
    combined_df = pd.concat(results_list, ignore_index=False)
    combined_df.to_csv("Data/demand_est.csv", index=True)


