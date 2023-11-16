"""Helper file for Thompson sampling"""

import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from tenacity import retry, stop_after_attempt, wait_fixed

import config as cfg

random.seed(42)

class ThompsonSampler:
    def __init__(self):
        self.placeholder = st.empty()

        self.latent_elasticity = cfg.LATENT_ELASTICITY
        self.price_observations = np.concatenate(
            [np.repeat(10,10), np.repeat(7.5,25), np.repeat(11,15)]
        )
        self.update_demand_observations()

        self.possible_prices = np.linspace(0, 20, 100)
        self.price_samples = []
        self.latent_demand = self.calc_latent_demand()
        self.latent_price = self.calc_optimal_price(self.latent_demand, sample=False)
        self.update_posteriors()

    def update_demand_observations(self):
        self.demand_observations = np.exp(
            np.random.normal(
                loc=-self.latent_elasticity*self.price_observations+cfg.LATENT_SHAPE,
                scale=cfg.LATENT_STDEV,
            )
        )

    def update_elasticity(self):
        self.latent_elasticity = st.session_state.latent_elasticity
        self.price_samples = []
        self.latent_demand = self.calc_latent_demand()
        self.update_demand_observations()
        self.latent_price = self.calc_optimal_price(self.latent_demand, sample=False)
        self.update_posteriors(samples=75)
        self.create_plots()

    def create_plots(self, highlighted_sample=None):
        with self.placeholder.container():
            posterior_plot, price_plot = st.columns(2)
            with posterior_plot:
                st.markdown("## Demands")
                fig = self.create_posteriors_plot(highlighted_sample)
                st.write(fig)
                plt.close(fig)
            with price_plot:
                st.markdown("## Prices")
                fig = self.create_price_plot()
                st.write(fig)
                plt.close(fig)

    def create_price_plot(self):
        fig = plt.figure()
        plt.xlabel("Price")
        plt.xlim(0,20)
        plt.yticks(color='w')

        price_distr = [self.calc_optimal_price(post_demand, sample=False)
                       for post_demand in self.posterior]
        plt.violinplot(price_distr, vert=False, showextrema=False)

        for price in self.price_samples:
            plt.plot(price, 1, marker='o', markersize = 5, color='grey', label="Price sample")

        plt.axhline(1, color='black')
        plt.axvline(self.latent_price, 0, color='red', label="Latent optimal price")

        _plot_legend()
        return fig

    def create_posteriors_plot(self, highlighted_sample=None):
        fig = plt.figure()
        plt.xlabel("Price")
        plt.ylabel("Demand")
        plt.xlim(0,20)
        plt.ylim(0,10)

        plt.scatter(self.price_observations, self.demand_observations, label="Demand observations")
        plt.plot(self.possible_prices, self.latent_demand, color="red", label="Latent demand")

        for posterior_sample in self.posterior_samples:
            plt.plot(
                self.possible_prices, posterior_sample,
                color="grey", alpha=0.15, label="Posterior demand"
            )
        if highlighted_sample is not None:
            plt.plot(
                self.possible_prices, highlighted_sample,
                color="black", label="Thompson sampled demand"
            )

        _plot_legend()
        return fig

    def calc_latent_demand(self):
        return np.exp(
            -self.latent_elasticity*self.possible_prices + cfg.LATENT_SHAPE
        )

    @staticmethod
    @np.vectorize
    def _cost(demand):
        return cfg.VARIABLE_COST*demand + cfg.FIXED_COST

    def calc_optimal_price(self, sampled_demand, sample=False):
        revenue = self.possible_prices * sampled_demand
        profit = revenue - self._cost(sampled_demand)
        optimal_price = self.possible_prices[np.argmax(profit)]
        if sample:
            self.price_samples.append(optimal_price)
            if len(self.price_samples) > cfg.MAX_PRICE_SAMPLES:
                self.price_samples.pop(0)
        return optimal_price

    def update_posteriors(self, samples=75):
        with open(f"assets/precalc_results/posterior_{self.latent_elasticity}.pkl", "rb") as post:
            self.posterior = pickle.load(post)
        self.posterior_samples = random.sample(self.posterior, samples)

    def pick_posterior(self):
        posterior_sample = random.choice(self.posterior_samples)
        self.calc_optimal_price(posterior_sample, sample=True)
        self.create_plots(highlighted_sample=posterior_sample)

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(0.25))
    def run(self):
        if st.session_state.latent_elasticity != self.latent_elasticity:
            self.update_elasticity()
        self.pick_posterior()

def _plot_legend():
    handles, labels = plt.gca().get_legend_handles_labels()
    labels, ids = np.unique(labels, return_index=True)
    handles = [handles[i] for i in ids]
    plt.legend(handles, labels, loc='upper right')
