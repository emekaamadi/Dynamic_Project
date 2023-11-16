"""Script that generates, pickles and stores posterior data"""

import pickle

import numpy as np

import config as cfg
from scripts.posterior import PosteriorGenerator

np.random.seed(42)

if __name__ == '__main__':
    demo_prices = np.concatenate([np.repeat(10,10), np.repeat(7.5,25), np.repeat(11,15)])
    possible_prices = np.linspace(0,20,100)

    for el in [x/100 for x in range(5,100,5)]:
        demo_demands = np.exp(
            np.random.normal(
                loc=-el*demo_prices+cfg.LATENT_SHAPE,
                scale=cfg.LATENT_STDEV,
            )
        )
        ts = PosteriorGenerator(prices=demo_prices, demands=demo_demands)
        posterior = ts.calc_posterior(samples=5000)
        post_demand_samples = []
        for idx in range(len(posterior)):
            elas = posterior.get_values("elas")[idx]
            shape = posterior.get_values("shape")[idx]
            post_demand_sample = np.exp(elas*possible_prices + shape)
            post_demand_samples.append(post_demand_sample)

        with open(f"assets/precalc_results/posterior_{el}.pkl", 'wb') as f:
            pickle.dump(post_demand_samples, f)
