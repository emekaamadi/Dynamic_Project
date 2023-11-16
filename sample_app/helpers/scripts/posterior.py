"""File for updating prior into posterior"""

import pymc3 as pm

class PosteriorGenerator:
    def __init__(self, prices, demands):
        self.price_observations = prices
        self.demand_observations = demands

    def calc_posterior(self, samples=1000):
        with pm.Model():
            elas = pm.Normal("elas",mu=-0.5, sd=0.5)
            shape = pm.Normal("shape",mu=0, sd=2)
            stdev = pm.Exponential("stdev",lam=1)
            y_hat = pm.math.dot(elas, self.price_observations) + shape
            log_observations = pm.math.log(self.demand_observations)
            _ = pm.Normal("demand", mu=y_hat, observed=log_observations, sigma=stdev)
            trace = pm.sample(samples)
        return trace
