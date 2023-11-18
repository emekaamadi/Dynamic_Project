import streamlit as st
import pandas as pd
from dynamic_pricing import DynamicPricing

# Streamlit 앱의 제목 설정
st.title('Dynamic Pricing Simulation for Cab Rides')

# 데이터 로드
df = pd.read_csv('cab_rides.csv')

# 사용자 입력 처리
st.header("1. Choose your options!")
product_id = st.selectbox('Select Service', df['product_id'].unique())
source = st.selectbox('Select Source Location', df['source'].unique())

# 사용자 입력에 기반한 데이터 필터링
filtered_data = df[(df['product_id'] == product_id) & (df['source'] == source)]

# Markdown에 설명 포함
st.header("\n2. Hyper-parameters to choose!")
st.markdown(
    """
    <style>
    .big-font {
        font-size:24px !important;
    }
    .text-red {
        color: red;
    }
    .text-blue {
        color: blue;
    }
    .text-green {
        color: green;
    }
    </style>

    <div class='big-font'>
        🙌  From now on, we'll be using the demand function below. <br>
        🙌  You can experiment by adjusting <span class='text-red'>a</span> and <span class='text-red'>b</span> in the demand function below. <br>
        🙌  <span class='text-blue'>Gamma (γ)</span> is a hyper-parameter that chooses what percentage to adjust the price by. Think of it as a similar concept to the step size in Gradient Descent. <br>
        🙌  <span class='text-green'>Eta (η)</span> is a demand elasticity parameter that is estimated by Thompson Sampling. <br><br><br><br>
    </div>
    """, 
    unsafe_allow_html=True
)
# st.markdown(r'''
#     $$a, b, \gamma, \eta$$
# ''', unsafe_allow_html=True)

# LaTeX를 사용하여 수요 함수 표시
demand_function_latex = r"Demand = a \times Price^{-|\eta|} + b"
st.latex(demand_function_latex)
st.markdown("where $a$ and $b$ are hyper-parameters and $\eta$ is estimated by Thompson Sampling.")


# DynamicPricing 인스턴스 생성을 위한 매개변수 입력
a = st.slider('$a$ Parameter', 0.0, 200.0, 100.0)
b = st.slider('$b$ Parameter', 0.0, 20.0, 10.0)
gamma = st.slider('Choose $\gamma$ Value (%)', 0.0, 1.0, 0.1)

# 시뮬레이션 실행 및 결과 표시
def run_simulation():
    # DynamicPricing 인스턴스 생성
    dynamic_pricing = DynamicPricing(data=filtered_data, a=a, b=b, gamma=gamma)
    dynamic_pricing.simulate_pricing_with_revenue_tracking()

    # 슬라이더를 통한 데이터 포인트 선택
    st.markdown("""<div><br><br></div>""", unsafe_allow_html=True)
    st.header("📈 Simulation Results")

    index_to_display = st.slider('Select Data Index', 0, len(dynamic_pricing.sampled_etas)-1, 0)

    # 선택된 데이터 인덱스에 따라 수익 계산
    dynamic_pricing.calculate_revenues(index_to_display)

    # 메시지 표시
    st.header("A. Summary:")
    message = dynamic_pricing.generate_message(index_to_display)
    st.text(message)

    # 수요 곡선 차트
    demand_curve_fig = dynamic_pricing.plot_demand_curve(index_to_display)
    st.header('B. Estimated Demand Curve')
    st.pyplot(demand_curve_fig)

    # 누적 수익 차트 표시
    st.header("C. Cumulative Revenues")
    cumulative_revenues_fig = dynamic_pricing.plot_cumulative_revenues(index_to_display)
    st.pyplot(cumulative_revenues_fig)

    # 수익 차이 차트 표시
    st.header("D. Accumulated Revenue Difference")
    revenue_difference_fig = dynamic_pricing.plot_revenue_difference(index_to_display)
    st.pyplot(revenue_difference_fig)

# 'Run Simulation' 버튼
if st.button('Run Simulation'):
    run_simulation()

# 슬라이더 값이 변경될 때마다 시뮬레이션 결과 업데이트
if 'dynamic_pricing' not in st.session_state or \
   (a != st.session_state.a or b != st.session_state.b or gamma != st.session_state.gamma):
    run_simulation()

# 현재 값들을 세션 상태에 저장
st.session_state.a = a
st.session_state.b = b
st.session_state.gamma = gamma









### Ver 3
# import streamlit as st
# import pandas as pd
# from dynamic_pricing import DynamicPricing

# # Streamlit 앱의 제목 설정
# st.title('Dynamic Pricing Simulation for Cab Rides')

# # 데이터 로드
# df = pd.read_csv('cab_rides.csv')

# # 사용자 입력 처리
# product_id = st.selectbox('Select Service', df['product_id'].unique(), key='product_id')
# source = st.selectbox('Select Source Location', df['source'].unique(), key='source')

# # 선택이 변경되었는지 확인하고, 변경된 경우 차트를 지웁니다.
# if 'prev_product_id' in st.session_state and 'prev_source' in st.session_state:
#     if st.session_state.prev_product_id != product_id or st.session_state.prev_source != source:
#         st.session_state.dynamic_pricing = None

# # 현재 선택을 저장하여 다음 실행에서 비교합니다.
# st.session_state.prev_product_id = product_id
# st.session_state.prev_source = source

# # 사용자 입력에 기반한 데이터 필터링
# filtered_data = df[(df['product_id'] == product_id) & (df['source'] == source)]

# # DynamicPricing 인스턴스 생성을 위한 매개변수 입력
# a = st.slider('a Parameter', 0.0, 200.0, 100.0)
# b = st.slider('b Parameter', 0.0, 20.0, 10.0)
# gamma = st.slider('Choose Gamma Value (%)', 0.0, 1.0, 0.1)

# # DynamicPricing 인스턴스 생성 및 시뮬레이션
# if st.button('Run Simulation', key='run_simulation_button'):
#     st.session_state.dynamic_pricing = DynamicPricing(data=filtered_data, a=a, b=b, gamma=gamma)
#     st.session_state.dynamic_pricing.simulate_pricing_with_revenue_tracking()
#     st.session_state.simulation_run = True

# # 슬라이더를 통한 데이터 포인트 선택 및 차트 표시
# if 'simulation_run' in st.session_state and st.session_state.simulation_run:
#     # 'dynamic_pricing' 객체가 있는지 확인
#     if hasattr(st.session_state, 'dynamic_pricing') and st.session_state.dynamic_pricing is not None:
#         index_to_display = st.slider('Select Data Index', 0, len(st.session_state.dynamic_pricing.sampled_etas)-1, 0)

#         # calculate_revenues 함수 호출
#         st.session_state.dynamic_pricing.calculate_revenues(index_to_display)
#         # 메시지 표시
#         message = st.session_state.dynamic_pricing.generate_message(index_to_display)
#         # st.markdown(f"<div style='padding: 10px; border: 1px solid black;'>{message}</div>", unsafe_allow_html=True)
#         st.text(message)
        
#         # 계산 중 메시지 표시
#         with st.spinner('Calculating...'):
#             # 선택된 데이터 인덱스에 따라 수익 계산
#             st.session_state.dynamic_pricing.calculate_revenues(index_to_display)

#             # 수요 곡선 차트
#             demand_curve_fig = st.session_state.dynamic_pricing.plot_demand_curve(index_to_display)
#             st.header('Estimated Demand Curve')
#             st.pyplot(demand_curve_fig)

#             # 누적 수익 차트 표시
#             st.header("Cumulative Revenues")
#             cumulative_revenues_fig = st.session_state.dynamic_pricing.plot_cumulative_revenues(index_to_display)
#             st.pyplot(cumulative_revenues_fig)

#             # 수익 차이 차트 표시
#             st.header("Revenue Difference")
#             revenue_difference_fig = st.session_state.dynamic_pricing.plot_revenue_difference(index_to_display)
#             st.pyplot(revenue_difference_fig)
# else:
#     st.write("Click 'Run Simulation' to generate the demand curve.")



# ### Ver 2
# # Streamlit 앱의 제목 설정
# st.title('Dynamic Pricing Simulation for Cab Rides')

# # 데이터 로드
# df = pd.read_csv('cab_rides.csv')

# # 사용자 입력 처리
# product_id = st.selectbox('Select Service', df['product_id'].unique(), key='product_id')
# source = st.selectbox('Select Source Location', df['source'].unique(), key='source')

# # 선택이 변경되었는지 확인하고, 변경된 경우 차트를 지웁니다.
# if 'prev_product_id' in st.session_state and 'prev_source' in st.session_state:
#     if st.session_state.prev_product_id != product_id or st.session_state.prev_source != source:
#         if 'dynamic_pricing' in st.session_state:
#             del st.session_state['dynamic_pricing']

# # 현재 선택을 저장하여 다음 실행에서 비교합니다.
# st.session_state.prev_product_id = product_id
# st.session_state.prev_source = source

# # 사용자 입력에 기반한 데이터 필터링
# filtered_data = df[(df['product_id'] == product_id) & (df['source'] == source)]

# # DynamicPricing 인스턴스 생성을 위한 매개변수 입력
# a = st.slider('a Parameter', 0.0, 200.0, 100.0)
# b = st.slider('b Parameter', 0.0, 20.0, 10.0)
# gamma = st.slider('Choose Gamma Value (%)', 0.0, 1.0, 0.1)

# # DynamicPricing 인스턴스 생성 및 시뮬레이션
# if st.button('Run Simulation', key='run_simulation_button'):
#     st.session_state.dynamic_pricing = DynamicPricing(data=filtered_data, a=a, b=b, gamma=gamma)
#     st.session_state.dynamic_pricing.simulate_pricing_with_revenue_tracking()
#     st.session_state.simulation_run = True

# # 슬라이더를 통한 데이터 포인트 선택 및 차트 표시
# if 'simulation_run' in st.session_state and st.session_state.simulation_run:
#     index_to_display = st.slider('Select Data Index', 0, len(st.session_state.dynamic_pricing.sampled_etas)-1, 0)
#     demand_curve_fig = st.session_state.dynamic_pricing.plot_demand_curve(index_to_display)
#     st.pyplot(demand_curve_fig)
# else:
#     st.write("Click 'Run Simulation' to generate the demand curve.")



### Ver 1
# import streamlit as st
# import pandas as pd
# import numpy as np
# from dynamic_pricing import DynamicPricing

# # Streamlit 앱의 제목 설정
# st.title('Dynamic Pricing Simulation for Cab Rides')

# # 데이터 로드
# df = pd.read_csv('cab_rides.csv')

# # 사용자 입력 처리
# product_id = st.selectbox('Select Service', df['product_id'].unique(), key='product_id')
# source = st.selectbox('Select Source Location', df['source'].unique(), key='source')

# # 선택이 변경되었는지 확인하고, 변경된 경우 차트를 지웁니다.
# if 'prev_product_id' in st.session_state and 'prev_source' in st.session_state:
#     if st.session_state.prev_product_id != product_id or st.session_state.prev_source != source:
#         if 'dynamic_pricing' in st.session_state:
#             del st.session_state['dynamic_pricing']

# # 현재 선택을 저장하여 다음 실행에서 비교합니다.
# st.session_state.prev_product_id = product_id
# st.session_state.prev_source = source

# # 사용자 입력에 기반한 데이터 필터링
# filtered_data = df[(df['product_id'] == product_id) & (df['source'] == source)]

# # DynamicPricing 인스턴스 생성을 위한 매개변수 입력
# a = st.slider('a Parameter', 0.0, 200.0, 100.0)
# b = st.slider('b Parameter', 0.0, 20.0, 10.0)
# gamma = st.slider('Choose Gamma Value (%)', 0.0, 1.0, 0.1)

# # DynamicPricing 인스턴스 생성 및 시뮬레이션
# if st.button('Run Simulation', key='run_simulation_button'):
#     st.session_state.dynamic_pricing = DynamicPricing(data=filtered_data, a=a, b=b, gamma=gamma)
#     st.session_state.dynamic_pricing.simulate_pricing_with_revenue_tracking()
#     st.session_state.simulation_run = True

#     # 슬라이더를 통한 데이터 포인트 선택
#     index_to_display = st.slider('Select Data Index', 0, len(st.session_state.dynamic_pricing.sampled_etas)-1, 0)

#     # 수요 곡선 차트 그리기
#     demand_curve_fig = st.session_state.dynamic_pricing.plot_demand_curve(index_to_display)
#     st.pyplot(demand_curve_fig)

# else:
#     st.write("Click 'Run Simulation' to generate the demand curve.")