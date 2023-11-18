# from dynamic_pricing import DynamicPricing
# import pandas as pd
# from ipywidgets import interact, IntSlider

# # 파일에서 필요한 데이터를 로드합니다. 예를 들면:
# FD_df = pd.read_csv('your_data_file.csv')  # 적절한 파일 이름으로 바꿔주세요.

# # DynamicPricing 클래스 인스턴스를 생성합니다.
# initial_eta_mean = 0.5
# initial_eta_std = 0.1
# a_value = 13  # 'a' 값 설정

# dp = DynamicPricing(initial_eta_mean, initial_eta_std, a_value)

# # 시뮬레이션을 실행합니다.
# # 예를 들어, DataFrame의 처음 2000개 행을 사용합니다.
# dp.simulate_pricing_with_revenue_tracking(FD_df[:2000])

# def interactive_chart(simulation):
#     """
#     Interactive chart 생성을 위한 함수입니다. 이 함수는 사용자의 입력에 따라 수요 곡선 차트를 업데이트합니다.

#     Parameters:
#     simulation (DynamicPricing): DynamicPricing 클래스의 인스턴스입니다.
#     """
#     def show_chart(index):
#         """
#         주어진 인덱스에 대한 수요 곡선을 보여줍니다.

#         Parameters:
#         index (int): 수요 곡선을 보여줄 인덱스입니다.
#         """
#         simulation.plot_demand_curve(index)
    
#     interact(show_chart, index=IntSlider(min=100, max=len(simulation.actual_prices), step=100, value=100))

# # 인터랙티브 차트 실행
# interactive_chart(dp)
