import pymc3 as pm
import theano.tensor as tt
import pandas as pd
import numpy as np
from preprocess import *


def load_MCMC_df(data = get_cleaned_data()):
    if "Unnamed: 0" in data.columns: 
        data.drop(columns=["Unnamed: 0"], axis=1, inplace=True)
    if "date_time" in data.columns: 
        data = data.set_index('date_time')
        data = data.sort_index()      
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
        eta = pm.Gamma('eta', alpha=eta_shape, beta=eta_rate)

        ### Set the priors for a and b
        a = pm.Uniform('a', lower=0, upper=10)
        b = pm.Uniform('b', lower=4, upper=70)

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
                    except Exception as e:
                        print(f"Error processing combination: CarType={car_type}, Source={source}, Destination={destination}")
                        print(f"Error message: {e}")

    # Combine all the results into a single dataframe
    combined_df = pd.concat(results_list, ignore_index=False)
    combined_df.to_csv("Data/demand_est_new.csv", index=True)