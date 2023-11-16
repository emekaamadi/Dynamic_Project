import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

class DynamicPricing:
    def __init__(self, initial_eta_mean, initial_eta_std, a, b):
        self.eta_mean = initial_eta_mean
        self.eta_std = initial_eta_std
        self.a = a
        self.b = b
        self.price_points = []
        self.eta_points = []
        self.sampled_etas = []
        self.actual_prices = []
        self.revenue_history = []
        self.cumulative_revenue = 0

    def update_posterior(self, price, actual_price, demand):
        if self.eta_std < 1e-6:
            self.eta_std = 1e-6

        eta_sample = -np.log(demand / self.a) / np.log(price)
        likelihood_std = 0.1

        new_mean_numerator = self.eta_mean / self.eta_std**2 + eta_sample / likelihood_std**2
        new_mean_denominator = 1 / self.eta_std**2 + 1 / likelihood_std**2
        new_mean = new_mean_numerator / new_mean_denominator if new_mean_denominator != 0 else self.eta_mean
        new_std = (1 / self.eta_std**2 + 1 / likelihood_std**2)**(-0.5) if new_mean_denominator != 0 else self.eta_std

        self.eta_mean = new_mean
        self.eta_std = max(new_std, 1e-6)

    def thompson_sampling(self):
        sampled_eta = np.random.normal(self.eta_mean, self.eta_std)
        self.sampled_etas.append(sampled_eta)
        return sampled_eta

    def get_demand(self, price, eta):
        return self.a * price ** (-np.abs(eta))

    def calculate_rmse(self):
        estimated_prices = [self.get_price(d, e) for d, e in zip(self.actual_prices, self.sampled_etas)]
        return np.sqrt(mean_squared_error(self.actual_prices, estimated_prices))

    def get_price(self, demand, eta):
        price = (demand / self.a) ** (-1 / np.abs(eta))
        self.price_points.append(price)
        return price

    def calculate_revenue(self, price, demand):
        return price * demand

    def simulate_pricing_with_revenue_tracking(self, data):
        for index, row in data.iterrows():
            sampled_eta = self.thompson_sampling()
            demand = self.get_demand(row['price'], sampled_eta)
            estimated_price = self.get_price(demand, sampled_eta)
            revenue = self.calculate_revenue(row['price'], demand)
            self.cumulative_revenue += revenue
            self.revenue_history.append(self.cumulative_revenue)

            self.actual_prices.append(row['price'])
            self.eta_points.append(sampled_eta)

            self.update_posterior(row['price'], estimated_price, demand)

            if index % 100 == 0 and index > 0:
                print(f'Index: {index}, RMSE: {self.calculate_rmse()}, Cumulative Revenue: {self.cumulative_revenue}, Eta: {self.eta_mean:.4f}')

    def simulate_pricing_with_revenue_estimation(self, price_range):
        estimated_revenues = []
        for price in price_range:
            # 비선형 수요 함수 예시: a / (price + b) 형태로 모델링
            # 여기서 a와 b는 모델 파라미터입니다.
            demand_estimate = self.a / (price + self.b)
            revenue_estimate = price * demand_estimate
            estimated_revenues.append(revenue_estimate)

        return price_range, estimated_revenues

    def plot_cumulative_revenue_over_time(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.revenue_history, label='Cumulative Revenue over time')
        plt.xlabel('Time')
        plt.ylabel('Cumulative Revenue')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_demand_curve(self, index):
        if index < 100 or index >= len(self.sampled_etas):
            print("Index out of range for plotting.")
            return
        plt.figure(figsize=(10, 6))
        plt.scatter(self.price_points[:index], [self.get_demand(p, e) for p, e in zip(self.price_points[:index], self.eta_points[:index])], color='gray', s=10)
        price_range = np.linspace(0.5 * min(self.price_points), 1.5 * max(self.price_points), 100)
        demand_curve = self.get_demand(price_range, self.sampled_etas[index-1])
        plt.plot(price_range, demand_curve, label=f'η: {self.sampled_etas[index-1]:.4f}')
        plt.xlabel('Price')
        plt.ylabel('Demand')
        plt.xlim([0, max(self.price_points)])
        plt.ylim([0, max([self.get_demand(p, self.eta_mean) for p in price_range])])
        plt.legend()
        plt.grid(True)
        plt.show()
