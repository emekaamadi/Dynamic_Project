import streamlit as st
import pandas as pd
from preprocess import * 
from predict import *
from util import *

########## Set Title ##########
st.title("Simulation for Ride Share Price Optimization")

########## Explain the problem ##########
st.header("Problem Explanation")
st.markdown("""
    <style>
    .big-font {
        font-size:24px !important;
    }
    .text-red {
        color: red;
        font-weight: bold;
        font-style: italic;
    }
    .text-blue {
        color: #87CEEB; /* Replace 'blue' with your chosen color */
        font-weight: bold;
        font-style: italic;
    }
    .text-green {
        color: #00FA9A; /* Replace 'green' with your chosen color */
        font-weight: bold;
        font-style: italic;
    }
    </style>

    <div class='big-font'>
        🙌  What is <span class='text-green'>dynamic pricing</span>? <br>
                - Basically, it's when companies adjust their prices based on the demand for their products or services. <br>
                - For example, if you check the price of a plane ticket at the airport <br>
                and then look at it again later, you'll see that the price is much higher. <br>
                Airlines do this to maximize revenue by raising prices <br>
                when there are fewer seats available <br>
                and more people want to buy tickets late. <br>
                Automating price changes with algorithms is called dynamic pricing. <br>
            <br>
        🙌  What is <span class='text-blue'>our team's goal?</span> <br>
                - Now we're going to explore how rideshare services <br>
                like Uber and Lyft use dynamic pricing with additional data <br>
                including weather, weekday/weekend. <br>
                - Our goal is to explore how basic pricing differs <br>
                from dynamic pricing and what factors influence this difference. <br>
                Uber and Lyft adjust their fares when demand is high by applying a value called a "surge multiplier". <br> 
                We'll also create our dynamic pricing using demand estimation. <br>
                By analyzing base pricing, dynamic pricing, and demand-based data, <br>
                we'll be able to see how prices change with demand and how dynamic pricing affects revenue. <br>
        <br>
    </div>
    """, 
    unsafe_allow_html=True
    )



########## Select Filters ##########
st.header("Select Options")
questions, answers = get_questions_answers()
car_type_options = get_service_types()

# Set filter options
options = []
for i in range(len(questions)):
    if i == 0:
        options.append(st.radio(questions[i], answers[i]))
    elif i == 3:
        if options[0] == 'Uber':
            options.append(st.selectbox(questions[i], car_type_options['Uber']))
        else:
            options.append(st.selectbox(questions[i], car_type_options['Lyft']))
    else:
        options.append(st.selectbox(questions[i], answers[i]))

# Make a button to create the dataframe
if st.button('Apply'):
    df = option_translator(options)
    st.session_state['df'] = df
    st.success('Successfully applied!')

    # # Show the dataframe
    # st.write("This is for testing purposes. Please ignore this.")
    # st.dataframe(df)



########## Prediction ##########
if 'df' in st.session_state:
    st.header("Model Predictions")
    st.markdown("""
    <style>
    .big-font {
        font-size:24px !important;
    }
    .text-red {
        color: red;
        font-weight: bold;
        font-style: italic;
    }
    .text-blue {
        color: #87CEEB; /* Replace 'blue' with your chosen color */
        font-weight: bold;
        font-style: italic;
    }
    .text-green {
        color: #00FA9A; /* Replace 'green' with your chosen color */
        font-weight: bold;
        font-style: italic;
    }
    </style>

    <div class='big-font'>
        🙌  <span class='text-green'>a</span> : The weights in the demand function. <br>
                The price elasticity of demand (ETA) is an indication of how much a change in price affects the quantity demanded.<br>
                The larger the value of A, the larger the change in quantity demanded for a small change in price, <br>
                indicating that the good is sensitive to price changes. <br>
        🙌  <span class='text-blue'>b</span> : This is the minimum demand. <br>
                B is responsible for shifting the demand curve in a vertical direction, <br>
                indicating that there will always be some amount of demand regardless of price.<br>
        <br>
    </div>
    """, 
    unsafe_allow_html=True
    )

    # Sliders for a and b values
    a = st.slider('Select value for a', min_value=1, max_value=200, value=100)
    b = st.slider('Select value for b', min_value=0.1, max_value=20.0, value=10.0)

    if st.button("Predict prices"):
        if not st.session_state['df'].empty:
            # Load models
            base_model, dynamic_model, demand_model = load_models()

            # Conduct predictions
            base_predictions, dynamic_predictions, demand_predictions = predict_prices(st.session_state['df'], base_model, dynamic_model, demand_model)
            base_predictions, dynamic_predictions, demand_predictions = adjust_demand_price(base_predictions[0], dynamic_predictions[0], demand_predictions[0])
            # Print the results
            st.markdown(f"<span style='color: white;'>Base Model Predictions:</span> <span style='color: green; font-size: 20px;'>{round(base_predictions, 4)}</span>", unsafe_allow_html=True)
            st.markdown(f"<span style='color: white;'>Dynamic Model Predictions:</span> <span style='color: green; font-size: 20px;'>{round(dynamic_predictions, 4)}</span>", unsafe_allow_html=True)
            st.markdown(f"<span style='color: white;'>Demand Model Predictions:</span> <span style='color: green; font-size: 20px;'>{round(demand_predictions, 4)}</span>", unsafe_allow_html=True)
            # add the predicted price to the dataframe
            st.session_state['df']['price'] = demand_predictions

            # ### If we don't predict eta
            # eta, a, b = get_estimated_values(MCMC_data=get_MCMC_data(), input_df=st.session_state['df'])

            ### If we predict eta
            eta = predict_eta(st.session_state['df'])
            st.session_state['df']['predicted_eta'] = eta

            if type(eta) == float:
                st.markdown(f"<span style='color: white;'>Estimated Demand:</span> <span style='color: green; font-size: 20px;'>{round(eta[0], 4)}</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"<span style='color: white;'>Estimated Demand:</span> <span style='color: green; font-size: 20px;'>{round(float(eta[0]), 4)}</span>", unsafe_allow_html=True)

    # Visualize Demand Function and Combined Chart button
    if st.button("Visualize Demand Function", key="visualize_demand"):
        if 'predicted_eta' in st.session_state['df']:
            # Load models
            base_model, dynamic_model, demand_model = load_models()

            # Conduct predictions
            base_predictions, dynamic_predictions, demand_predictions = predict_prices(st.session_state['df'], base_model, dynamic_model, demand_model)
            base_predictions, dynamic_predictions, demand_predictions = adjust_demand_price(base_predictions[0], dynamic_predictions[0], demand_predictions[0])

            # Show demand function
            plt_demand_func = plot_demand_func(st.session_state['df'], a, b)
            st.pyplot(plt_demand_func)
            
            # Show combined chart
            plt_combined = plot_revenue_bar_chart(st.session_state['df'], base_predictions, dynamic_predictions, demand_predictions, a, b)
            st.pyplot(plt_combined)
            
        else:
            st.warning("No eta predictions available. Please predict prices first.")