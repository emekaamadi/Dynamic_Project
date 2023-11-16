import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def plot_cumulative_revenue(dynamic_pricing_obj):
    """
    Plot the cumulative revenue over time using the data from the DynamicPricing object.

    Parameters:
    dynamic_pricing_obj (DynamicPricing): An instance of the DynamicPricing class.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(dynamic_pricing_obj.revenue_history, label='Cumulative Revenue over time')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Revenue')
    plt.title('Cumulative Revenue Over Time')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

def plot_demand_curve(dynamic_pricing_obj, index):
    """
    Plot the demand curve for a specific index.

    Parameters:
    dynamic_pricing_obj (DynamicPricing): An instance of the DynamicPricing class.
    index (int): The index to plot the demand curve for.
    """
    if index < 0 or index >= len(dynamic_pricing_obj.sampled_etas):
        st.write("Index out of range for plotting.")
        return

    plt.figure(figsize=(10, 6))
    plt.scatter(dynamic_pricing_obj.price_points[:index], [dynamic_pricing_obj.get_demand(p, e) for p, e in zip(dynamic_pricing_obj.price_points[:index], dynamic_pricing_obj.eta_points[:index])], color='gray', s=10)
    price_range = np.linspace(0.5 * min(dynamic_pricing_obj.price_points), 1.5 * max(dynamic_pricing_obj.price_points), 100)
    demand_curve = dynamic_pricing_obj.get_demand(price_range, dynamic_pricing_obj.sampled_etas[index-1])
    plt.plot(price_range, demand_curve, label=f'η: {dynamic_pricing_obj.sampled_etas[index-1]:.4f}')
    plt.xlabel('Price')
    plt.ylabel('Demand')
    plt.title('Demand Curve')
    plt.xlim([0, max(dynamic_pricing_obj.price_points)])
    plt.ylim([0, max([dynamic_pricing_obj.get_demand(p, dynamic_pricing_obj.eta_mean) for p in price_range])])
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

def plot_price_vs_cumulative_revenue(dynamic_pricing_obj, price_range):
    prices, revenues = dynamic_pricing_obj.simulate_pricing_with_revenue_estimation(price_range)
    cumulative_revenues = np.cumsum(revenues)  # 누적 수익 계산

    plt.figure(figsize=(10, 6))
    plt.plot(prices, cumulative_revenues, label='Cumulative Revenue vs Price')
    plt.xlabel('Price')
    plt.ylabel('Cumulative Revenue')
    plt.title('Price vs Cumulative Revenue')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

def plot_results(dynamic_pricing_obj, index):
    """
    Plot results using the data from the DynamicPricing object.

    Parameters:
    dynamic_pricing_obj (DynamicPricing): An instance of the DynamicPricing class.
    index (int): The index to plot the demand curve for.
    """
    plot_cumulative_revenue(dynamic_pricing_obj)
    plot_demand_curve(dynamic_pricing_obj, index)
