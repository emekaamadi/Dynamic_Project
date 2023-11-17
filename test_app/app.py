import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dynamic_pricing import DynamicPricing


# Streamlit 앱의 제목 설정
st.title('Dynamic Pricing Simulation \nfor Cab Rides')

# 데이터 로드
df = pd.read_csv('cab_rides.csv')

# 사용자 입력 처리
product_id = st.selectbox('Select Service', df['product_id'].unique())
source = st.selectbox('Select Source Location', df['source'].unique())

# 사용자 입력에 기반한 데이터 필터링
filtered_data = df[(df['product_id'] == product_id) & (df['source'] == source)]

# DynamicPricing 인스턴스 생성을 위한 매개변수 입력
a = st.number_input('a Parameter', value=100.0)
b = st.number_input('b Parameter', value=10.0)

# DynamicPricing 인스턴스 생성 및 시뮬레이션
if 'dynamic_pricing' not in st.session_state or st.button('Run Simulation'):
    st.session_state.dynamic_pricing = DynamicPricing(data=filtered_data, a=a, b=b)
    st.session_state.dynamic_pricing.simulate_pricing_with_revenue_tracking()

# 슬라이더 및 차트 표시
if 'dynamic_pricing' in st.session_state:
    # 슬라이더를 통한 데이터 포인트 선택
    index_to_display = st.slider('Select Data Index', 0, len(st.session_state.dynamic_pricing.sampled_etas)-1, 0)

    # 계산 중임을 나타내는 placeholder 생성
    placeholder_demand_curve = st.empty()
    placeholder_revenue_curve = st.empty()

    # 수요 곡선 차트 그리기
    demand_curve_fig = st.session_state.dynamic_pricing.plot_demand_curve(index_to_display)
    # 계산 중 메시지 표시
    placeholder_demand_curve.text("Calculating the demand curve...")
    # placeholder를 실제 수요 곡선 차트로 대체
    placeholder_demand_curve.pyplot(demand_curve_fig)

    # 예상 매출 차트 그리기
    price_range = np.linspace(0.01, 20, 200)  # 예상 매출 차트를 위한 가격 범위
    revenue_fig, optimal_price, max_revenue = st.session_state.dynamic_pricing.plot_demand_and_revenue(price_range)
    # placeholder를 실제 예상 매출 차트로 대체
    placeholder_revenue_curve.pyplot(revenue_fig)
    st.write(f"Optimal Price: {optimal_price:.2f}, Max Revenue: {max_revenue:.2f}")
