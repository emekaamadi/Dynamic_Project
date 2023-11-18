import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import config

class DynamicPricing:
    def __init__(self, data, a, b, gamma=0.1):
        self.data = data.copy()
        self.a = a
        self.b = b
        self.gamma = gamma
        self.eta_mean = config.LATENT_ELASTICITY
        self.eta_std = config.LATENT_STDEV
        self.sampled_etas = []
        self.actual_prices = list(self.data['price'])
        self.revenue_history = []
        self.revenue_difference = []
    
    def get_demand(self, price, eta):
        eta = np.clip(np.abs(eta), 0.1, 0.7) * np.sign(eta)
        return self.a * price ** (-np.abs(eta)) + self.b
    
    def thompson_sampling(self):
        # 현실적인 범위 내에서 eta 값을 샘플링
        sampled_eta = np.random.normal(self.eta_mean, self.eta_std)
        sampled_eta = np.clip(sampled_eta, -1, 1)  # 예시: 샘플링된 eta 값을 -1과 1 사이로 제한
        self.sampled_etas.append(sampled_eta)
        return sampled_eta
    
    def simulate_pricing_with_revenue_tracking(self):
        for price in self.actual_prices:
            sampled_eta = self.thompson_sampling()
            demand = self.get_demand(price, sampled_eta)
            revenue = price * demand
            self.revenue_history.append(revenue)
    
    def plot_demand_curve(self, index_to_display):
        fig, ax = plt.subplots(figsize=(10, 6))
        price_range = np.linspace(0.5 * min(self.actual_prices), 1.5 * max(self.actual_prices), 100)

        # 이전 수요 곡선을 회색으로 투명하게 그리기
        for past_index in range(index_to_display):
            past_demand_curve = [self.get_demand(p, self.sampled_etas[past_index]) for p in price_range]
            ax.plot(price_range, past_demand_curve, color='grey', alpha=0.1)

        # 과거 가격 데이터 포인트를 빨간색으로 누적하여 그리기
        ax.scatter(self.actual_prices[:index_to_display], [self.get_demand(price, self.sampled_etas[past_index]) for past_index, price in enumerate(self.actual_prices[:index_to_display])], color='red', alpha=0.5)

        # 현재 선택된 인덱스에 해당하는 수요 곡선을 진한 파란색으로 그리기
        current_demand_curve = [self.get_demand(p, self.sampled_etas[index_to_display]) for p in price_range]
        ax.plot(price_range, current_demand_curve, color='blue', alpha=1.0, label=f'η: {self.sampled_etas[index_to_display]:.4f}')

        ax.set_xlabel('Price')
        ax.set_ylabel('Demand')
        ax.set_xlim([0, 1.5 * max(self.actual_prices)])
        ax.set_ylim([0, max(current_demand_curve) * 1.5])
        ax.legend()
        ax.grid(True)
        return fig

    def calculate_revenues(self, index_to_display):
        self.optimized_revenue = []
        self.actual_accumulated_revenue = []
        self.revenue_difference = []  # 최적화된 수익과 실제 수익의 차이 리스트
        cumulative_optimized_revenue = 0
        cumulative_actual_revenue = 0

        for idx in range(index_to_display + 1):
            price = self.actual_prices[idx]
            sampled_eta = self.sampled_etas[idx]
            demand = self.get_demand(price, sampled_eta)

            if abs(sampled_eta) > 0.28:
                optimal_price = price * (1 - self.gamma)
            else:
                optimal_price = price * (1 + self.gamma)

            optimized_demand = self.get_demand(optimal_price, sampled_eta)
            optimized_revenue = optimal_price * optimized_demand
            cumulative_optimized_revenue += optimized_revenue
            self.optimized_revenue.append(cumulative_optimized_revenue)

            actual_revenue = price * demand
            cumulative_actual_revenue += actual_revenue
            self.actual_accumulated_revenue.append(cumulative_actual_revenue)

            # 최적화된 수익과 실제 수익의 차이 계산 및 저장
            revenue_diff = cumulative_optimized_revenue - cumulative_actual_revenue
            self.revenue_difference.append(revenue_diff)


    def plot_cumulative_revenues(self, index_to_display):
        self.calculate_revenues(index_to_display)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.optimized_revenue, label='Optimized Revenue')
        ax.plot(self.actual_accumulated_revenue, label='Actual Accumulated Revenue')
        ax.set_xlabel('Data Points')
        ax.set_ylabel('Cumulative Revenue (USD)')
        ax.legend(loc='upper left')
        ax.grid(True)
        return fig

    def plot_revenue_difference(self, index_to_display):
        if index_to_display >= len(self.revenue_difference):
            return "Selected index is out of range."

        revenue_difference = self.revenue_difference[index_to_display]

        self.calculate_revenues(index_to_display)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.revenue_difference, label='Revenue Difference', color='green')
        ax.set_xlabel('Data Points')
        ax.set_ylabel('Revenue Difference (USD)')
        ax.legend()
        ax.grid(True)
        return fig
    
    def generate_message(self, index_to_display):
        sampled_eta = self.sampled_etas[index_to_display]
        actual_price = self.actual_prices[index_to_display]
        optimal_price = actual_price * (1 - self.gamma if abs(sampled_eta) > 0.28 else 1 + self.gamma)
        revenue_difference = self.revenue_difference[index_to_display]

        message = (
            f"Current η (Eta): {sampled_eta:.4f} \n"
            f"Actual Price: ${actual_price:.2f} \n"
            f"Optimized Price: ${optimal_price:.2f} (Gamma: {self.gamma:.2f}) \n"
            f"Cumulative Revenue Difference (Optimized vs. Actual): ${revenue_difference:.2f}"
        )
        return message

    # def plot_revenue_comparison(self, index_to_display):
    #     self.calculate_revenues(index_to_display)  # 슬라이더에 따른 수익 계산

    #     fig, ax1 = plt.subplots(figsize=(10, 6))
    #     ax1.plot(self.optimized_revenue, label='Optimized Revenue')
    #     ax1.plot(self.actual_accumulated_revenue, label='Actual Accumulated Revenue')
    #     ax1.set_xlabel('Data Points')
    #     ax1.set_ylabel('Cumulative Revenue (USD)')
    #     ax1.legend(loc='upper left')
    #     ax1.grid(True)

    #     # 추가 y축 생성하여 수익 차이 표시
    #     ax2 = ax1.twinx()
    #     ax2.plot(self.revenue_difference, label='Revenue Difference', color='green')
    #     ax2.set_ylabel('Revenue Difference (USD)')
    #     ax2.legend(loc='upper right')

    #     return fig
    
    # def calculate_optimized_revenue(self):
    #     self.optimized_revenue = []
    #     self.actual_accumulated_revenue = []
    #     self.revenue_difference = []  # 최적화된 수익과 실제 수익의 차이 리스트
    #     cumulative_optimized_revenue = 0
    #     cumulative_actual_revenue = 0

    #     for price in self.actual_prices:
    #         sampled_eta = self.thompson_sampling()
    #         demand = self.get_demand(price, sampled_eta)

    #         if abs(sampled_eta) > 0.28:
    #             optimal_price = price * (1 - self.gamma)
    #         else:
    #             optimal_price = price * (1 + self.gamma)

    #         optimized_demand = self.get_demand(optimal_price, sampled_eta)
    #         optimized_revenue = optimal_price * optimized_demand
    #         cumulative_optimized_revenue += optimized_revenue
    #         self.optimized_revenue.append(cumulative_optimized_revenue)

    #         actual_revenue = price * demand
    #         cumulative_actual_revenue += actual_revenue
    #         self.actual_accumulated_revenue.append(cumulative_actual_revenue)

    #         # 최적화된 수익과 실제 수익의 차이 계산 및 저장
    #         revenue_diff = cumulative_optimized_revenue - cumulative_actual_revenue
    #         self.revenue_difference.append(revenue_diff)

    # def plot_revenue_comparison(self):
    #     fig, ax1 = plt.subplots(figsize=(10, 6))
    #     ax1.plot(self.optimized_revenue, label='Optimized Revenue')
    #     ax1.plot(self.actual_accumulated_revenue, label='Actual Accumulated Revenue')
    #     ax1.set_xlabel('Data Points')
    #     ax1.set_ylabel('Cumulative Revenue (USD)')
    #     ax1.legend(loc='upper left')
    #     ax1.grid(True)

    #     # 추가 y축 생성하여 수익 차이 표시
    #     ax2 = ax1.twinx()
    #     ax2.plot(self.revenue_difference, label='Revenue Difference', color='green')
    #     ax2.set_ylabel('Revenue Difference (USD)')
    #     ax2.legend(loc='upper right')

    #     return fig








### Ver 2
# import numpy as np
# import matplotlib.pyplot as plt
# import streamlit as st
# import pandas as pd
# import pymc3 as pm
# import theano.tensor as tt
# import config

# class DynamicPricing:
#     def __init__(self, data, a, b, gamma=0.1):
#         self.data = data.copy()
#         self.a = a
#         self.b = b
#         self.gamma = gamma
#         self.eta_mean = config.LATENT_ELASTICITY
#         self.eta_std = config.LATENT_STDEV
#         self.sampled_etas = []
#         self.actual_prices = list(self.data['price'])
#         self.revenue_history = []

#     def get_demand(self, price, eta):
#         # price가 Theano 텐서가 아닌 경우 변환
#         if not isinstance(price, tt.TensorVariable):
#             price = tt.as_tensor_variable(price)
#         return self.a * tt.pow(price, -tt.abs_(eta)) + self.b

#     def update_eta_distribution(self):
#         if 'demand' not in self.data.columns:
#             self.data['demand'] = self.calculate_demand(self.data)

#         with pm.Model() as model:
#             eta_prior = pm.Normal('eta', mu=self.eta_mean, sd=self.eta_std)
#             likelihood = pm.Normal('likelihood', mu=self.get_demand(self.data['price'], eta_prior), sd=0.1, observed=self.data['demand'])
#             trace = pm.sample(1000, return_inferencedata=False)

#         self.eta_mean = np.mean(trace['eta'])
#         self.eta_std = np.std(trace['eta'])

#     def calculate_demand(self, data):
#         with pm.Model() as model:
#             eta = pm.Normal('eta', mu=self.eta_mean, sd=self.eta_std)
#             trace = pm.sample(1000, return_inferencedata=False)

#         demand_samples = [self.get_demand(data['price'].values, eta_sample).eval() for eta_sample in trace['eta']]
#         return np.mean(demand_samples, axis=0)

#     def thompson_sampling(self):
#         sampled_eta = np.random.normal(self.eta_mean, self.eta_std)
#         self.sampled_etas.append(sampled_eta)
#         return sampled_eta

#     def simulate_pricing_with_revenue_tracking(self):
#         for price in self.actual_prices:
#             sampled_eta = self.thompson_sampling()
#             demand = self.get_demand(price, sampled_eta).eval()
#             revenue = price * demand
#             self.revenue_history.append(revenue)

#     def plot_demand_curve(self, index_to_display):
#         fig, ax = plt.subplots(figsize=(10, 6))
#         price_range = np.linspace(min(self.actual_prices), max(self.actual_prices), 100)

#         sampled_eta = self.sampled_etas[index_to_display]
#         demand_curve = [self.get_demand(price, sampled_eta).eval() for price in price_range]
#         ax.plot(price_range, demand_curve, color='blue')
#         ax.set_xlabel('Price')
#         ax.set_ylabel('Demand')
#         ax.grid(True)
#         plt.show()
#         return fig






### Ver 1
# import numpy as np
# import matplotlib.pyplot as plt
# import streamlit as st
# import pandas as pd
# import pymc3 as pm
# import theano.tensor as tt
# import config as cfg

# class DynamicPricing:
#     def __init__(self, data, a, b, eta_mean=None, eta_std=None, gamma=0.1):
#         self.data = data.copy()  # DataFrame의 복사본을 사용합니다.
#         self.eta_mean = eta_mean if eta_mean is not None else 0.5
#         self.eta_std = eta_std if eta_std is not None else 0.1
#         self.a = a
#         self.b = b
#         self.price_points = []
#         self.eta_points = []
#         self.sampled_etas = []
#         self.actual_prices = []
#         self.revenue_history = []
#         self.cumulative_revenue = 0
#         self.update_demand_observations()
#         self.gamma = gamma

#     def get_demand(self, price, eta):
#         # 주어진 가격과 eta 값에 따라 수요를 계산합니다.
#         return self.a * price ** (-np.abs(eta)) + self.b
#         # return self.a / (price ** np.abs(eta) + self.b)

#     def get_price(self, demand, eta):
#         # 주어진 수요와 eta 값에 따라 가격을 계산합니다.
#         price = (demand / self.a) ** (-1 / np.abs(eta))
#         self.price_points.append(price)
#         return price

#     def update_eta_distribution(self):
#         # 데이터에 따라 수요를 계산하는 로직 구현
#         self.data['demand'] = self.get_demand_from_data(self.data)

#         with pm.Model() as model:
#             # 사전 분포 정의
#             eta_prior = pm.Normal('eta', mu=self.eta_mean, sd=self.eta_std)

#             # 관찰된 데이터와 모델 연결
#             likelihood = pm.Normal('likelihood', mu=self.model_demand_function(eta_prior), sd=적절한_표준편차, observed=observed_demand)

#             # MCMC를 사용하여 사후 분포 샘플링
#             trace = pm.sample(1000, return_inferencedata=False)

#         # 사후 분포에서 eta 값을 업데이트
#         self.eta_mean = np.mean(trace['eta'])
#         self.eta_std = np.std(trace['eta'])

#     def get_demand_from_data(self, data):
#         # 데이터로부터 수요를 계산하는 로직
#         # 예시: 각 행에 대해 get_demand 메서드를 호출
#         return [self.get_demand(row['price'], self.eta_mean) for index, row in data.iterrows()]

#     def model_demand_function(self, eta):
#         # PyMC3/Theano에서 사용할 수 있는 Theano 텐서 연산을 사용하여 수요 계산
#         # 가격 데이터를 Theano 텐서로 변환 (예: self.data['price'].values를 Theano 텐서로 변환)
#         price_tensor = tt.as_tensor_variable(self.data['price'].values)

#         # Theano 텐서 연산을 사용하여 수요 계산
#         # 예: self.a * price ** (-abs(eta)) + self.b
#         demand = self.a * tt.pow(price_tensor, -tt.abs_(eta)) + self.b
#         return demand

#     def thompson_sampling(self):
#         # Thompson Sampling을 사용하여 eta 값을 샘플링합니다.
#         sampled_eta = np.random.normal(self.eta_mean, self.eta_std)
#         self.sampled_etas.append(sampled_eta)
#         return sampled_eta

#     def simulate_pricing_with_revenue_tracking(self):
#         for index, row in self.data.iterrows():
#             sampled_eta = self.thompson_sampling()
#             demand = self.get_demand(row['price'], sampled_eta)
#             # estimated_price = self.get_price(demand, sampled_eta)
#             revenue = row['price'] * demand
#             self.cumulative_revenue += revenue
#             self.revenue_history.append(self.cumulative_revenue)
#             self.actual_prices.append(row['price'])
#             self.eta_points.append(sampled_eta)


#     # def plot_cumulative_revenue_over_time(self):
#     #     fig, ax = plt.subplots(figsize=(10, 6))
#     #     ax.plot(self.revenue_history, label='Cumulative Revenue over time')
#     #     ax.set_xlabel('Time')
#     #     ax.set_ylabel('Cumulative Revenue')
#     #     ax.legend()
#     #     ax.grid(True)
#     #     return fig  # fig 객체를 반환


#     def calculate_expected_revenue(self, price):
#         demand = self.get_demand(price, self.eta_mean)
#         return demand * price
    

#     def calc_optimal_price(self, current_price):
#         # 현재 가격에 대한 수요와 eta 추정
#         current_demand = self.get_demand(current_price, self.eta_mean)
#         current_revenue = current_price * current_demand
#         current_eta = self.eta_mean  # 현재 데이터 포인트의 eta

#         # eta에 따른 가격 조정
#         if current_eta > 0.5:
#             # 가격을 낮추는 것이 좋을 때
#             adjusted_price = current_price * (1 - self.gamma)
#         else:
#             # 가격을 높이는 것이 좋을 때
#             adjusted_price = current_price * (1 + self.gamma)

#         # 조정된 가격에 대한 수요와 매출 추정
#         adjusted_demand = self.get_demand(adjusted_price, current_eta)
#         adjusted_revenue = adjusted_price * adjusted_demand

#         # 최적 가격 결정
#         if adjusted_revenue > current_revenue:
#             optimal_price = adjusted_price
#             max_revenue = adjusted_revenue
#         else:
#             optimal_price = current_price
#             max_revenue = current_revenue

#         return optimal_price, max_revenue


#     def plot_demand_and_revenue(self, price_range):
#         # 가격에 따른 수요와 매출을 계산
#         demands = [self.get_demand(price, self.eta_mean) for price in price_range]
#         revenues = [price * demand for price, demand in zip(price_range, demands)]

#         # 최적 가격과 최대 매출 계산
#         optimal_price, max_revenue = self.calc_optimal_price(price_range)
        
#         # 차트 그리기
#         fig, ax1 = plt.subplots(figsize=(10, 6))

#         # 수요 곡선 그리기
#         ax1.plot(price_range, demands, 'b-', label='Demand')
#         ax1.set_xlabel('Price')
#         ax1.set_ylabel('Demand', color='b')
#         ax1.tick_params('y', colors='b')

#         # 매출 곡선을 같은 그래프에 그리기 위한 두 번째 y축 추가
#         ax2 = ax1.twinx()
#         ax2.plot(price_range, revenues, 'r-', label='Revenue')
#         ax2.set_ylabel('Revenue', color='r')
#         ax2.tick_params('y', colors='r')

#         # 최적 가격 선 그리기
#         ax2.axvline(x=optimal_price, color='g', linestyle='--', label=f'Optimal Price: {optimal_price:.2f}')

#         # 범례 및 제목 설정
#         ax1.legend(loc='upper left')
#         ax2.legend(loc='upper right')
#         ax1.set_title('Demand and Revenue vs. Price')

#         plt.grid(True)
#         plt.show()

#         return fig, optimal_price, max_revenue
    

#     def plot_demand_curve(self, index_to_display):
#         fig, ax = plt.subplots(figsize=(10, 6))
#         price_range = np.linspace(0.5 * min(self.price_points), 1.5 * max(self.price_points), 100)

#         # 이전 수요 곡선을 회색으로 투명하게 그리기
#         for past_index in range(100, index_to_display):
#             past_demand_curve = [self.get_demand(p, self.sampled_etas[past_index-1]) for p in price_range]
#             ax.plot(price_range, past_demand_curve, color='grey', alpha=0.1)  # 낮은 알파 값으로 거의 투명하게 표시

#         # 과거의 가격 데이터 포인트를 빨간색으로 누적하여 그리기
#         for past_index in range(100, index_to_display):
#             ax.scatter(self.price_points[past_index], self.get_demand(self.price_points[past_index], self.sampled_etas[past_index]), color='red', alpha=0.5)

#         # 현재 선택된 인덱스에 해당하는 수요 곡선을 진한 파란색으로 그리기
#         current_demand_curve = [self.get_demand(p, self.sampled_etas[index_to_display-1]) for p in price_range]
#         ax.plot(price_range, current_demand_curve, color='blue', alpha=1.0, label=f'η: {self.sampled_etas[index_to_display-1]:.4f}')  # 진한 파란색으로 표시

#         ax.set_xlabel('Price')
#         ax.set_ylabel('Demand')
#         ax.set_xlim([0, 1.5 * max(self.price_points)]) 
#         ax.set_ylim([0, 1.5 * max(current_demand_curve)])
#         ax.legend()
#         ax.grid(True)

#         return fig

#     ### Test
#     def calculate_confidence_interval(self, optimal_price):
#         # 신뢰 구간 계산 로직 구현
#         # 예시: 최적 가격 주변의 수요 변동성을 분석하여 신뢰 구간 계산
#         lower_bound = optimal_price * 0.9  # 예시 계산
#         upper_bound = optimal_price * 1.1  # 예시 계산
#         return lower_bound, upper_bound

#     def adjust_price_based_on_gamma(self, current_price):
#         adjusted_price_up = current_price * (1 + self.gamma)
#         adjusted_price_down = current_price * (1 - self.gamma)
#         return adjusted_price_up, adjusted_price_down

#     def plot_optimal_price_recommendation(self, current_price):
#         # 현재 데이터 포인트에서의 가격 및 eta 사용
#         current_eta = self.sampled_etas[-1]

#         # 현재 가격에 대한 예상 수익 계산
#         current_demand = self.get_demand(current_price, current_eta)
#         current_revenue = current_price * current_demand

#         # 최적 가격 계산
#         optimal_price, max_revenue = self.calc_optimal_price(current_price)

#         # 차트 그리기
#         fig, ax = plt.subplots()
#         ax.plot([current_price, optimal_price], [current_revenue, max_revenue], label='Revenue Comparison', marker='o')

#         # 현재 가격 및 최적 가격 표시
#         ax.axvline(x=current_price, color='blue', linestyle='--', label=f'Current Price: {current_price:.2f}')
#         ax.axvline(x=optimal_price, color='green', linestyle='--', label=f'Optimal Price: {optimal_price:.2f}')

#         ax.set_xlabel('Price')
#         ax.set_ylabel('Revenue')
#         ax.legend()
#         ax.grid(True)
#         plt.show()

#         return fig

#     def calculate_optimal_price_based_on_elasticity(self):
#         # 시장 수용 가능한 최소 및 최대 가격 설정
#         min_price = min(self.actual_prices)
#         max_price = max(self.actual_prices)

#         # 가격 범위에 대한 수요 및 매출 계산
#         price_range = np.linspace(min_price, max_price, 100)
#         demand_estimates = [self.get_demand(price, self.eta_mean) for price in price_range]
#         revenue_estimates = [price * demand for price, demand in zip(price_range, demand_estimates)]

#         # 최대 매출을 생성하는 가격 찾기
#         max_revenue = max(revenue_estimates)
#         optimal_price = price_range[revenue_estimates.index(max_revenue)]

#         return optimal_price, max_revenue

#     # 추천 메시지 생성
#     def generate_recommendation_message(self, current_price):
#         # 시장 수용 가능한 최소 및 최대 가격 설정
#         min_price = min(self.actual_prices)
#         max_price = max(self.actual_prices)

#         # 가격 범위 설정
#         price_range = np.linspace(min_price, max_price, 100)

#         # 최적 가격 계산
#         optimal_price, _ = self.calc_optimal_price(price_range)

#         # 추천 메시지 생성
#         message = f"Current Price: ${current_price}, Optimal Price: ${optimal_price:.2f}. "
#         message += "Consider adjusting the price to optimize revenue."
#         return message