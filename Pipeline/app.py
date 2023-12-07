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
    }
    .text-blue {
        color: blue;
    }
    .text-green {
        color: green;
    }
    </style>

    <div class='big-font'>
        🙌  Here we will add some explanation of the purpose of this app. <br>
        🙌  This section can use colors to highlight important points like <span class='text-red'>a</span> <br>
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

    # Show the dataframe
    st.write("This is for testing purposes. Please ignore this.")
    st.dataframe(df)



########## Prediction ##########
if 'df' in st.session_state:
    st.header("Model Predictions")
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
            plt_combined = plot_combined_charts(st.session_state['df'], base_predictions, dynamic_predictions, demand_predictions, a, b)
            st.pyplot(plt_combined)
            
        else:
            st.warning("No eta predictions available. Please predict prices first.")