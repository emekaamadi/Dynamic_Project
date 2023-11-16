"""Streamlit entrypoint"""

import time

import numpy as np
import streamlit as st

from helpers.thompson_sampling import ThompsonSampler

np.random.seed(42)

st.set_page_config(
    page_title="Dynamic Pricing",
    page_icon="💸",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items={
        'Get help': None,
        'Report a bug': None,
        'About': "https://www.ml6.eu/",
    }
)

st.title("Dynamic Pricing")
st.subheader("Setting optimal prices with Bayesian stats 📈")

st.markdown("""In this demo you will see  \n
👉 How Bayesian demand function estimates are created based on sales data  \n
👉 How Thompson sampling will generate concrete price points from these Bayesian estimates  \n
👉 The impact of price elasticity on Bayesian demand estimation""")
st.markdown("""You will notice:  \n
👉 As you increase price elasticity, the demand becomes more sensitive to price changes and thus the
profit-optimizing price becomes lower (& vice versa).  \n
👉 As you decrease price elasticity, our demand observations at €7.5, €10 and €11 become
increasingly larger and increasingly more variable (as their variance is a constant fraction of the
absolute value). This causes our demand posterior to become increasingly wider and thus Thompson
sampling will lead to more exploration.
""")
st.markdown("""If you are looking for more insights into how dynamic pricing is done in practice,
check out our blog post here: https://medium.com/ml6team/dynamic-pricing-in-practice-99fe2216a93d""")

thompson_sampler = ThompsonSampler()
demo_button = st.checkbox(
    label='Ready for the Demo? 🕹️',
    help="Starts interactive Thompson sampling demo"
)
elasticity = st.slider(
    "Adjust latent elasticity",
    key="latent_elasticity",
    min_value=0.05,
    max_value=0.95,
    value=0.25,
    step=0.05,
)
while demo_button:
    thompson_sampler.run()
    time.sleep(1)
