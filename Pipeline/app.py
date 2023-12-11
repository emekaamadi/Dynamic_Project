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
            <br>
        🙌  What is <span class='text-blue'>our team's goal?</span> <br>
                - Now we're going to explore how rideshare services <br>
                like Uber and Lyft use dynamic pricing with additional data <br>
                - By analyzing base pricing, dynamic pricing, and demand-based data, <br>
                we'll be able to see how prices change with demand and how dynamic pricing affects revenue <br>
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
    # Set the flag to reset sliders
    st.session_state['reset_sliders'] = True
    st.rerun()



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
        🙌  <span class='text-green'>a</span> : The weights in the demand elasiticity <br>
        🙌  <span class='text-blue'>b</span> : This is the minimum demand <br>
        <br>
    </div>
    """, 
    unsafe_allow_html=True
    )

    # Initialize or reset the session state for sliders
    if 'reset_sliders' in st.session_state and st.session_state['reset_sliders']:
        st.session_state['a_value'] = 0.5
        st.session_state['b_value'] = 30.0
        st.session_state['reset_sliders'] = False

    # Sliders for a and b
    a = st.slider('Select value for a', min_value=0.1, max_value=10.0, value=st.session_state.get('a_value', 0.5))
    b = st.slider('Select value for b', min_value=4.0, max_value=50.0, value=st.session_state.get('b_value', 30.0))

    # Save the current slider values to session state
    st.session_state['a_value'] = a
    st.session_state['b_value'] = b

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

            ### Markdown for the equation
            st.markdown("""
                    <style>
                    .big-font {
                        font-size:24px !important;
                    }
                    </style>

                    <div class='big-font'>
                        🙌  Revenue = Demand * Price <br>
                        <br>
                    </div>
                    """, 
                    unsafe_allow_html=True
                    )           

            # Show revenue bar chart
            plt_combined = plot_revenue_bar_chart(st.session_state['df'], base_predictions, dynamic_predictions, demand_predictions, a, b)
            st.pyplot(plt_combined)

        else:
            st.warning("No eta predictions available. Please predict prices first.")