# This file is going to be used for charting and visualizing the data.
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import altair as alt
from predict import *
from demand_estimation import *


def plot_demand_func(df, a=100, b=10):
    # Set the style
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


# def plot_combined_charts(df, base_price, dynamic_price, demand_price, a=100, b=10):
#     # Calculate expected revenue and demand
#     base_revenue, dynamic_revenue, demand_revenue = calculate_revenue(df, base_price, dynamic_price, demand_price, a, b)
#     base_demand = a * base_price ** (-df['predicted_eta'].iloc[0]) + b
#     dynamic_demand = a * dynamic_price ** (-df['predicted_eta'].iloc[0]) + b
#     demand_demand = a * demand_price ** (-df['predicted_eta'].iloc[0]) + b
    
#     # Create data for visualization of prices and expected revenue
#     strategies = ['Base', 'Dynamic', 'Demand']
#     prices = [base_price, dynamic_price, demand_price]
#     demands = [base_demand, dynamic_demand, demand_demand]
#     revenues = [base_revenue, dynamic_revenue, demand_revenue]

#     # Set the figure size
#     fig, ax1 = plt.subplots(figsize=(10, 9))

#     # Calculate the max and min for both demand and revenue
#     all_demands = demands + prices  # Combine demands and prices for primary y-axis scaling
#     all_revenues = revenues
#     max_demand = max(all_demands)
#     max_revenue = max(all_revenues)
#     min_demand = min(all_demands)
#     min_revenue = min(all_revenues)

#     # Add padding to max and min for better visualization
#     demand_padding = (max_demand - min_demand) * 0.1
#     revenue_padding = (max_revenue - min_revenue) * 0.1

#     # Set y-axis limits for demand
#     ax1.set_ylim(min_demand - demand_padding, max_demand + demand_padding)
#     ax1.bar(strategies, demands, color='navy', label='Demand', alpha=0.2, width=0.2)
#     ax1.set_xlabel('Pricing Strategies')
#     ax1.set_ylabel('Demand & Price', color='navy')
#     ax1.tick_params(axis='y', labelcolor='navy')

#     # Scatter plot for prices
#     ax1.scatter(strategies, prices, color='darkgreen', label='Price', zorder=5)
#     for i, strategy in enumerate(strategies):
#         ax1.text(i, prices[i], f'{prices[i]:.2f}', ha='center', va='bottom', color='darkgreen', fontweight='bold')

#     # Set y-axis limits for revenue
#     ax2 = ax1.twinx()
#     ax2.set_ylim(min_revenue - revenue_padding, max_revenue + revenue_padding)
#     ax2.plot(strategies, revenues, color='darkred', label='Revenue', marker='o', linestyle='-', linewidth=2)
#     ax2.set_ylabel('Revenue', color='darkred')
#     ax2.tick_params(axis='y', labelcolor='darkred')

#     # Annotate the actual numbers on top of the bars and lines
#     for i, strategy in enumerate(strategies):
#         ax1.text(i, demands[i], f'{demands[i]:.2f}', ha='center', va='bottom', color='navy', fontweight='bold')
#         ax2.text(i, revenues[i], f'{revenues[i]:.2f}', ha='center', va='top', color='darkred', fontweight='bold')

#     # Combine legends from both axes
#     handles1, labels1 = ax1.get_legend_handles_labels()
#     handles2, labels2 = ax2.get_legend_handles_labels()
#     ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper left')

#     plt.title('Combined Chart for Demand, Price, and Revenue')
#     plt.show()
#     return fig








