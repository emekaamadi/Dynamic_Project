import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import config as cfg

class DynamicPricing:
    def __init__(self, data, a, b, eta_mean=None, eta_std=None):
        self.data = data.copy()  # DataFrame의 복사본을 사용합니다.
        self.eta_mean = eta_mean if eta_mean is not None else 0.5
        self.eta_std = eta_std if eta_std is not None else 0.1
        self.a = a
        self.b = b
        self.price_points = []
        self.eta_points = []
        self.sampled_etas = []
        self.actual_prices = []
        self.revenue_history = []
        self.cumulative_revenue = 0
        self.update_demand_observations()

    def update_demand_observations(self):
        # 데이터셋에서 수요 데이터를 처리하는 로직을 구현할 수 있습니다.
        self.data['demand'] = self.data['distance'] * self.a  # 거리에 비례하는 수요로 가정

    def thompson_sampling(self):
        # Thompson Sampling을 사용하여 eta 값을 샘플링합니다.
        sampled_eta = np.random.normal(self.eta_mean, self.eta_std)
        self.sampled_etas.append(sampled_eta)
        return sampled_eta

    def get_demand(self, price, eta):
        # 주어진 가격과 eta 값에 따라 수요를 계산합니다.
        return self.a * price ** (-np.abs(eta))
    # def get_demand(self, price, eta):
    #     # 가격이 증가하면 수요가 감소하는 형태의 수요 함수
    #     return self.a / (price ** np.abs(eta) + self.b)

    def simulate_pricing_with_revenue_tracking(self):
        for index, row in self.data.iterrows():
            sampled_eta = self.thompson_sampling()
            demand = self.get_demand(row['price'], sampled_eta)
            estimated_price = self.get_price(demand, sampled_eta)
            revenue = row['price'] * demand
            self.cumulative_revenue += revenue
            self.revenue_history.append(self.cumulative_revenue)
            self.actual_prices.append(row['price'])
            self.eta_points.append(sampled_eta)

    def get_price(self, demand, eta):
        # 주어진 수요와 eta 값에 따라 가격을 계산합니다.
        price = (demand / self.a) ** (-1 / np.abs(eta))
        self.price_points.append(price)
        return price

    def plot_cumulative_revenue_over_time(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.revenue_history, label='Cumulative Revenue over time')
        ax.set_xlabel('Time')
        ax.set_ylabel('Cumulative Revenue')
        ax.legend()
        ax.grid(True)
        return fig  # fig 객체를 반환

    def calculate_expected_revenue(self, price):
        demand = self.get_demand(price, self.eta_mean)
        return demand * price
    
    def calc_optimal_price(self, price_range):
        # 가격 범위에 따른 예상 매출 계산
        revenues = []
        for price in price_range:
            # 여기서 get_demand 함수는 실제 수요 데이터를 기반으로 한 추정치를 반환해야 합니다.
            estimated_demand = self.get_demand(price, self.eta_mean)  # 현재 eta_mean을 사용하여 추정된 수요를 계산
            revenue = price * estimated_demand  # 가격과 추정된 수요를 곱하여 매출을 계산
            revenues.append(revenue)
        
        # 최대 매출과 해당하는 최적 가격 계산
        max_revenue = max(revenues)
        optimal_price_index = revenues.index(max_revenue)
        optimal_price = price_range[optimal_price_index]
        
        return optimal_price, max_revenue
    
    def plot_demand_and_revenue(self, price_range):
        # 가격에 따른 수요와 매출을 계산
        demands = [self.get_demand(price, self.eta_mean) for price in price_range]
        revenues = [price * demand for price, demand in zip(price_range, demands)]

        # 최적 가격과 최대 매출 계산
        optimal_price, max_revenue = self.calc_optimal_price(price_range)
        
        # 차트 그리기
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # 수요 곡선 그리기
        ax1.plot(price_range, demands, 'b-', label='Demand')
        ax1.set_xlabel('Price')
        ax1.set_ylabel('Demand', color='b')
        ax1.tick_params('y', colors='b')

        # 매출 곡선을 같은 그래프에 그리기 위한 두 번째 y축 추가
        ax2 = ax1.twinx()
        ax2.plot(price_range, revenues, 'r-', label='Revenue')
        ax2.set_ylabel('Revenue', color='r')
        ax2.tick_params('y', colors='r')

        # 최적 가격 선 그리기
        ax2.axvline(x=optimal_price, color='g', linestyle='--', label=f'Optimal Price: {optimal_price:.2f}')

        # 범례 및 제목 설정
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax1.set_title('Demand and Revenue vs. Price')

        plt.grid(True)
        plt.show()

        return fig, optimal_price, max_revenue
    
    def plot_demand_curve(self, index_to_display):
        fig, ax = plt.subplots(figsize=(10, 6))
        price_range = np.linspace(0.5 * min(self.price_points), 1.5 * max(self.price_points), 100)

        # 이전 수요 곡선을 회색으로 투명하게 그리기
        for past_index in range(100, index_to_display):
            past_demand_curve = [self.get_demand(p, self.sampled_etas[past_index-1]) for p in price_range]
            ax.plot(price_range, past_demand_curve, color='grey', alpha=0.1)  # 낮은 알파 값으로 거의 투명하게 표시

        # 과거의 가격 데이터 포인트를 빨간색으로 누적하여 그리기
        for past_index in range(100, index_to_display):
            ax.scatter(self.price_points[past_index], self.get_demand(self.price_points[past_index], self.sampled_etas[past_index]), color='red', alpha=0.5)

        # 현재 선택된 인덱스에 해당하는 수요 곡선을 진한 파란색으로 그리기
        current_demand_curve = [self.get_demand(p, self.sampled_etas[index_to_display-1]) for p in price_range]
        ax.plot(price_range, current_demand_curve, color='blue', alpha=1.0, label=f'η: {self.sampled_etas[index_to_display-1]:.4f}')  # 진한 파란색으로 표시

        ax.set_xlabel('Price')
        ax.set_ylabel('Demand')
        ax.set_xlim([0, 17.5])
        ax.set_ylim([0, 140])
        ax.legend()
        ax.grid(True)

        return fig