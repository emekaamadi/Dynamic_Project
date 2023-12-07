# This file is going to be used for charting and visualizing the data.
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import altair as alt
from predict import *

def plot_demand_func(df, a=100, b=10):
    eta = predict_eta(df)  
    df['predicted_eta'] = eta
    x = np.linspace(2, 40, 100) 
    y = (a * x ** (-eta) + b)  

    plt.plot(x, y, label='Demand Curve')

    actual_demand = a * df['price'] ** (-df['predicted_eta']) + b

    plt.scatter(df['price'], actual_demand, color='green', label='Current Prices')

    plt.xlabel('Price')
    plt.ylabel('Demand')
    plt.title('Demand Function')
    plt.legend()
    return plt

def calculate_revenue(df, base_price, dynamic_price, demand_price, a=100, b=10):
    if 'predicted_eta' not in df:
        eta = predict_eta(df)  # 'predict_eta' 함수는 eta 값을 예측하는 함수입니다.
        df['predicted_eta'] = eta
    else:
        eta = df['predicted_eta'].iloc[0]

    # 각 가격 전략에 대한 수요 계산
    base_demand = a * base_price ** (-eta) + b
    dynamic_demand = a * dynamic_price ** (-eta) + b
    demand_demand = a * demand_price ** (-eta) + b

    # 예상 매출 계산
    base_revenue = base_price * base_demand
    dynamic_revenue = dynamic_price * dynamic_demand
    demand_revenue = demand_price * demand_demand

    return base_revenue, dynamic_revenue, demand_revenue

# def plot_combined_charts(df, base_price, dynamic_price, demand_price, a=100, b=10):
#     # Calculate expected revenue
#     base_revenue, dynamic_revenue, demand_revenue = calculate_revenue(df, base_price, dynamic_price, demand_price, a, b)
#     # Calculate expected demand
#     base_demand = a * base_price ** (-df['predicted_eta'].iloc[0]) + b
#     dynamic_demand = a * dynamic_price ** (-df['predicted_eta'].iloc[0]) + b
#     demand_demand = a * demand_price ** (-df['predicted_eta'].iloc[0]) + b
    
#     # Create data for visualization of prices and expected revenue
#     strategies = ['Base', 'Dynamic', 'Demand']
#     prices = [base_price, dynamic_price, demand_price]
#     demands = [base_demand, dynamic_demand, demand_demand]
#     revenues = [base_revenue, dynamic_revenue, demand_revenue]

#     fig, ax1 = plt.subplots(figsize=(10, 9))

#     # Dynamically adjust the y-axis for demand to fit the data
#     demand_max = max(demands)
#     demand_min = min(demands)
#     demand_padding = demand_max * 0.1  # Add 10% padding above the max
#     ax1.set_ylim(demand_min - demand_padding, demand_max + demand_padding)

#     # Bar chart for demands with thinner bars and darker color
#     ax1.bar(strategies, demands, color='navy', label='Demand', alpha=0.6, width=0.4)
#     ax1.set_xlabel('Pricing Strategies')
#     ax1.set_ylabel('Demand', color='navy')
#     ax1.tick_params(axis='y', labelcolor='navy')

#     # Scatter plot for prices with annotation
#     for i, strategy in enumerate(strategies):
#         ax1.scatter(strategy, prices[i], color='darkgreen', label='Price' if i == 0 else "", zorder=5)
#         ax1.text(strategy, prices[i], f'{prices[i]:.2f}', ha='center', va='bottom', color='darkgreen', fontweight='bold')

#     # Dynamically adjust the y-axis for revenue to fit the data
#     ax2 = ax1.twinx()
#     revenue_max = max(revenues)
#     revenue_padding = revenue_max * 0.1  # Add 10% padding above the max
#     ax2.set_ylim(0, revenue_max + revenue_padding)
#     ax2.plot(strategies, revenues, color='darkred', label='Revenue', marker='o', linestyle='-', linewidth=2)
#     ax2.set_ylabel('Revenue', color='darkred')
#     ax2.tick_params(axis='y', labelcolor='darkred')

#     # Annotate the actual numbers on top of the bars and lines
#     for i, strategy in enumerate(strategies):
#         ax1.text(i, demands[i], f'{demands[i]:.2f}', ha='center', va='bottom', color='navy', fontweight='bold')
#         # Adjust the position of revenue annotations to be above the line
#         ax2.text(i, revenues[i] + revenue_padding * 0.1, f'{revenues[i]:.2f}', ha='center', va='bottom', color='darkred', fontweight='bold')

#     # Adding a legend that combines both axes
#     handles1, labels1 = ax1.get_legend_handles_labels()
#     handles2, labels2 = ax2.get_legend_handles_labels()
#     ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper left')

#     plt.title('Combined Chart for Demand, Price, and Revenue')
#     plt.show()
#     return fig

def plot_combined_charts(df, base_price, dynamic_price, demand_price, a=100, b=10):
    # Calculate expected revenue and demand
    base_revenue, dynamic_revenue, demand_revenue = calculate_revenue(df, base_price, dynamic_price, demand_price, a, b)
    base_demand = a * base_price ** (-df['predicted_eta'].iloc[0]) + b
    dynamic_demand = a * dynamic_price ** (-df['predicted_eta'].iloc[0]) + b
    demand_demand = a * demand_price ** (-df['predicted_eta'].iloc[0]) + b
    
    # Create data for visualization of prices and expected revenue
    strategies = ['Base', 'Dynamic', 'Demand']
    prices = [base_price, dynamic_price, demand_price]
    demands = [base_demand, dynamic_demand, demand_demand]
    revenues = [base_revenue, dynamic_revenue, demand_revenue]

    # Set the figure size
    fig, ax1 = plt.subplots(figsize=(10, 9))

    # Calculate the max and min for both demand and revenue
    all_demands = demands + prices  # Combine demands and prices for primary y-axis scaling
    all_revenues = revenues
    max_demand = max(all_demands)
    max_revenue = max(all_revenues)
    min_demand = min(all_demands)
    min_revenue = min(all_revenues)

    # Add padding to max and min for better visualization
    demand_padding = (max_demand - min_demand) * 0.1
    revenue_padding = (max_revenue - min_revenue) * 0.1

    # Set y-axis limits for demand
    ax1.set_ylim(min_demand - demand_padding, max_demand + demand_padding)
    ax1.bar(strategies, demands, color='navy', label='Demand', alpha=0.2, width=0.2)
    ax1.set_xlabel('Pricing Strategies')
    ax1.set_ylabel('Demand & Price', color='navy')
    ax1.tick_params(axis='y', labelcolor='navy')

    # Scatter plot for prices
    ax1.scatter(strategies, prices, color='darkgreen', label='Price', zorder=5)
    for i, strategy in enumerate(strategies):
        ax1.text(i, prices[i], f'{prices[i]:.2f}', ha='center', va='bottom', color='darkgreen', fontweight='bold')

    # Set y-axis limits for revenue
    ax2 = ax1.twinx()
    ax2.set_ylim(min_revenue - revenue_padding, max_revenue + revenue_padding)
    ax2.plot(strategies, revenues, color='darkred', label='Revenue', marker='o', linestyle='-', linewidth=2)
    ax2.set_ylabel('Revenue', color='darkred')
    ax2.tick_params(axis='y', labelcolor='darkred')

    # Annotate the actual numbers on top of the bars and lines
    for i, strategy in enumerate(strategies):
        ax1.text(i, demands[i], f'{demands[i]:.2f}', ha='center', va='bottom', color='navy', fontweight='bold')
        ax2.text(i, revenues[i], f'{revenues[i]:.2f}', ha='center', va='top', color='darkred', fontweight='bold')

    # Combine legends from both axes
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper left')

    plt.title('Combined Chart for Demand, Price, and Revenue')
    plt.show()
    return fig








