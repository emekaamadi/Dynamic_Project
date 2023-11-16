import streamlit as st
import pandas as pd
import numpy as np
from data_loader import load_and_filter_data
from dynamic_pricing import DynamicPricing
from utils import plot_cumulative_revenue, plot_demand_curve, plot_price_vs_cumulative_revenue

# Streamlit 앱 타이틀 설정
st.title('Dynamic Pricing Simulation for Lyft Rides')

# 'Lyft'를 기반으로 데이터 로드 및 필터링
df = load_and_filter_data('cab_rides.csv', 'Lyft')

# 사용자 입력을 통한 product_id 및 source 선택
product_id = st.selectbox('Select Lyft Service', df['product_id'].unique())
source = st.selectbox('Select Source Location', df['source'].unique())

# 사용자 입력에 기반한 데이터 필터링
filtered_data = df[(df['product_id'] == product_id) & (df['source'] == source)]

# 동적 가격 책정 시뮬레이션을 위한 사용자 입력
initial_eta_mean = st.number_input('Initial η Mean', value=1.0)
initial_eta_std = st.number_input('Initial η Standard Deviation', value=1.0)
a = st.number_input('a Parameter', value=1.0)
b = st.number_input('b Parameter', value=1.0)  # 'b' 매개변수 입력 추가

# DynamicPricing 객체 생성
dynamic_pricing = DynamicPricing(initial_eta_mean, initial_eta_std, a, b)

# 시뮬레이션 실행 버튼
if st.button('Run Simulation'):
    dynamic_pricing.simulate_pricing_with_revenue_tracking(filtered_data)

    # 가격 범위 입력 받기
    min_price = st.number_input('Minimum Price', value=0.0, step=0.1)
    max_price = st.number_input('Maximum Price', value=10.0, step=0.1)
    price_step = st.number_input('Price Step', value=0.1, step=0.1)

    # 시뮬레이션 버튼이 눌렸을 때 실행
    if min_price < max_price and price_step > 0:
        price_range = np.arange(min_price, max_price, price_step)
        plot_price_vs_cumulative_revenue(dynamic_pricing, price_range)
    else:
        st.error("Please ensure that the max price is greater than the min price and the step is positive.")
        
    # 기존 수요 곡선 차트 유지
    index = st.number_input('Select Index for Demand Curve', min_value=0, max_value=len(dynamic_pricing.actual_prices), step=1, value=0)
    if index < len(dynamic_pricing.actual_prices):
        plot_demand_curve(dynamic_pricing, index)

# 나머지 Streamlit 코드...
