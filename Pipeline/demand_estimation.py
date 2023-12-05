import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pymc3 as pm
import theano.tensor as tt
from preprocess import get_cleaned_data


def estimate_demand_parameters(dataframe, price_col):
    """
    Estimates the demand parameters (eta, a, b) using PyMC3 and adds them as new columns in the dataframe.

    :param dataframe: Pandas DataFrame containing the data
    :param price_col: Name of the column containing price data
    :return: DataFrame with new columns for eta, a, b estimates
    """
    with pm.Model() as model:
        # Define priors
        # eta에 대한 감마 분포 정의, 평균이 0.4~0.5가 되도록 조정
        eta_shape = 2   # 모양 매개변수
        eta_rate = 5    # 비율 매개변수
        eta = pm.Gamma('eta', alpha=eta_shape, beta=eta_rate) + 0.25
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

def save_demand_data(data):
    data = get_cleaned_data()

    lst_car_type = list(set(data.car_type))
    lst_source = list(set(data.source))
    lst_destination = list(set(data.destination))

    results_list = []

    # 모든 조합에 대해 반복
    for car_type in lst_car_type:
        for source in lst_source:
            for destination in lst_destination:
                # 데이터 필터링
                filtered_data = data[(data['car_type'] == car_type) & (data['source'] == source) & (data['destination'] == destination)]

                # 필터링된 데이터가 비어있지 않은 경우에만 처리
                if not filtered_data.empty:
                    try:
                        # 함수 호출
                        result_df, _ = estimate_demand_parameters(filtered_data, 'price')
                        # 결과 데이터프레임을 리스트에 추가
                        results_list.append(result_df)
                    except Exception as e:
                        print(f"Error processing combination: CarType={car_type}, Source={source}, Destination={destination}")
                        print(f"Error message: {e}")

    # 모든 결과 데이터프레임을 하나로 결합
    combined_df = pd.concat(results_list, ignore_index=True)
    combined_df.to_csv("Data/demand_est.csv", index=False)

    
