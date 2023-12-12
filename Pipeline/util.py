# This file is going to be used for charting and visualizing the data.
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import altair as alt
import os
from scipy.stats import gamma
from predict import *
from demand_estimation import *
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from preprocess import *
from train import *


def plot_demand_func(df, a=100, b=10):
    # Set the style
    a = a * 500
    plt.style.use('fivethirtyeight')

    # Calculate the predicted prices and demands
    base_predictions, dynamic_predictions, demand_predictions = predict_prices(df, load_models()[0], load_models()[1], load_models()[2])
    base_predictions, dynamic_predictions, demand_predictions = adjust_demand_price(base_predictions[0], dynamic_predictions[0], demand_predictions[0])

    # Calculate demand values for the predicted prices
    eta = predict_eta(df)  
    df['predicted_eta'] = eta
    base_demand = a * base_predictions ** (-df['predicted_eta']) + b
    dynamic_demand = a * dynamic_predictions ** (-df['predicted_eta']) + b
    demand_demand = a * demand_predictions ** (-df['predicted_eta']) + b

    # Plot the price points first with a higher zorder to ensure they are above the demand curve
    plt.scatter(base_predictions, base_demand, color='darkolivegreen', label='Price of Base Model', zorder=3)
    plt.scatter(dynamic_predictions, dynamic_demand, color='darkorange', label='Price of Dynamic Model', zorder=3)
    plt.scatter(demand_predictions, demand_demand, color='darkred', label='Price of Demand Model', zorder=3)

    # Now plot the demand curve, with a lower zorder to ensure it's below the price points
    x = np.linspace(2, 40, 100) 
    y = a * x ** (-eta) + b
    plt.figsize=(10, 6)
    plt.plot(x, y, label='Demand Curve', zorder=2)

    plt.xlabel('Price', fontsize=14)
    plt.ylabel('Demand', fontsize=14)
    plt.title('Demand Function', fontweight='bold', fontsize=20)
    plt.legend(fontsize=14)
    
    return plt


def calculate_revenue(df, base_price, dynamic_price, demand_price, a=100, b=10):
    if 'predicted_eta' not in df:
        eta = predict_eta(df)
        df['predicted_eta'] = eta
    else:
        eta = df['predicted_eta'].iloc[0]

    # Calculate the demand for each strategy
    base_demand = a * base_price ** (-eta) + b
    dynamic_demand = a * dynamic_price ** (-eta) + b
    demand_demand = a * demand_price ** (-eta) + b

    # Calculate the revenue for each strategy
    base_revenue = base_price * base_demand
    dynamic_revenue = dynamic_price * dynamic_demand
    demand_revenue = demand_price * demand_demand

    return base_revenue, dynamic_revenue, demand_revenue


def plot_revenue_bar_chart(df, base_price, dynamic_price, demand_price, a=100, b=10):
    # Set the style
    a = a * 500
    plt.style.use('fivethirtyeight')

    # Calculate expected revenue
    base_revenue, dynamic_revenue, demand_revenue = calculate_revenue(df, base_price, dynamic_price, demand_price, a, b)
    
    # Create data for visualization of expected revenue
    strategies = ['Base', 'Dynamic', 'Demand']
    revenues = [base_revenue, dynamic_revenue, demand_revenue]
    colors = ['darkolivegreen', 'darkorange', 'darkred']  # Colors for each bar

    # Set the figure size, adjust the width for more space
    fig, ax = plt.subplots(figsize=(10, 6))  # Increased width

    # Plot the revenue data with a thinner bar width and different colors
    bar_width = 0.5  # Adjust the width as needed
    for i, (strategy, revenue, color) in enumerate(zip(strategies, revenues, colors)):
        ax.bar(strategy, revenue, color=color, label=f'{strategy} Revenue', width=bar_width)
        # Annotate the actual numbers on top of the bars with the same color
        ax.text(i, revenue, f'{revenue:.2f}', ha='center', va='bottom', color=color, fontweight='bold', fontsize=14)

    ax.set_xlabel('Pricing Strategies', fontsize=14)
    ax.set_ylabel('Revenue', fontsize=14)
    
    # Place the legend outside of the figure/plot
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Revenue Types')

    plt.title('Estimated Revenue for Pricing Strategies\n', fontweight='bold', fontsize=20)
    plt.tight_layout()  # Adjust the padding between and around subplots.
    plt.show()
    
    return fig


def compare_model_predictions(data):
    # Just to get surge indices IGNORE
    A = data.drop(['price'], axis=1)
    b = data['price']
    X_train_pre, X_valid_pre, y_train_pre, y_valid_pre = train_test_split(A, b, train_size=0.8, test_size=0.2, random_state=0)
    surge_indices = X_valid_pre['surge_multiplier'].values > 1


    # Prepare the data
    columns = ["cab_type", "source", "destination", "car_type", "weekday", "rush_hour", "is_raining", "temp_groups","price"]
    data = data[columns]

    X,y, _ = prepare_data(data)
    
    # Split the data
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

    # Load the models
    base_model = load(f'Models/base_model_pipeline.joblib')
    dynamic_model = load(f'Models/dynamic_model_pipeline.joblib')

    # Filter X_valid and y_valid for surge > 1
    X_valid_surge = X_valid[surge_indices]
    y_valid_surge = y_valid.values[surge_indices]

    # Predict using both models
    base_predictions = base_model.predict(X_valid_surge)
    dynamic_predictions = dynamic_model.predict(X_valid_surge)

    # Plotting the whole Test Set as a Scatter Plot
    plt.figure(figsize=(12, 6))
    
    plt.scatter(range(len(y_valid_surge)), y_valid_surge, label='Actual Prices', color='blue', marker='o', s=5)
    plt.scatter(range(len(y_valid_surge)), base_predictions, label='Base Model Predictions', color='green', marker='x', s=5)
    plt.scatter(range(len(y_valid_surge)), dynamic_predictions, label='Dynamic Model Predictions', color='red', marker='+', s=5)
    plt.title('Comparison of Actual Prices vs Predictions (Surge > 1)')
    plt.xlabel('Sample Index')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig('Visuals/plot.png')
    plt.show()

    #  the first 100 Values of the Test Set for Visual Clarity
    plt.figure(figsize=(12, 6))
    plt.plot(y_valid_surge[:100], label='Actual Prices', color='blue', marker='o')
    plt.plot(base_predictions[:100], label='Base Model Predictions', color='green', marker='x')
    plt.plot(dynamic_predictions[:100], label='Dynamic Model Predictions', color='red', marker='+')
    plt.title('Comparison of Actual Prices vs Predictions (Surge > 1) First 100 Values')
    plt.xlabel('Sample Index')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig('Visuals/plot2.png')
    plt.show()

def plot_gamma_distribution(shape=2, rate=5, filename='gamma_distribution'):
    # Generate values
    x = np.linspace(0, 1.5, 1000)
    y = gamma.pdf(x, shape, scale=1/rate)

    # Plot
    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label=f'Shape={shape}, Rate={rate}')
    plt.title('Gamma Distribution')
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True)

    # Save plot to file
    if not os.path.exists('Visuals'):
        os.makedirs('Visuals')
    plt.savefig(f'Visuals/{filename}.png', format='png')
    plt.close()

def plot_demand_function(a=10, b=40, eta_fixed=0.4, shape=2, rate=5, filename='demand_function'):
    # Originally it should be done with trace object from pymc3
    # But the restriction of the project is to use only numpy and scipy
    # Price range
    price_range = np.linspace(2.5, 40, 100)

    # Gamma distribution for eta
    eta_random = gamma.rvs(shape, scale=1/rate, size=300)

    # Plotting the demand curves
    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(8,6))

    # 200 random eta values to create grey lines with transparency
    for eta in eta_random:
        demand = a * price_range ** (-eta) + b
        plt.plot(price_range, demand, color='grey', alpha=0.1)

    # Fixed eta value to create the blue line
    demand_fixed = a * price_range ** (-eta_fixed) + b
    plt.plot(price_range, demand_fixed, label='Estimated Demand Function', color='blue', linewidth=2)

    # Finalizing the plot
    plt.title('Demand Function Estimations')
    plt.xlabel('Price')
    plt.ylabel('Demand')
    plt.legend()
    plt.grid(True)

    # Save plot to file
    if not os.path.exists('Visuals'):
        os.makedirs('Visuals')
    plt.savefig(f'Visuals/{filename}.png', format='png')
    plt.close()




