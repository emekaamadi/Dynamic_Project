import streamlit as st
import pandas as pd
from preprocess import get_service_types, get_questions_answers, option_translator
from predict import load_models, predict_prices

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
    st.header("2. Model Predictions")
    if st.button("Predict prices"):
        if not st.session_state['df'].empty:
            # Load models
            base_model, dynamic_model, demand_model = load_models()

            # Conduct predictions
            base_predictions, dynamic_predictions, demand_predictions = predict_prices(st.session_state['df'], base_model, dynamic_model, demand_model)

            # Print the results
            st.markdown(f"<span style='color: white;'>Base Model Predictions:</span> <span style='color: green; font-size: 20px;'>{round(base_predictions[0], 4)}</span>", unsafe_allow_html=True)
            st.markdown(f"<span style='color: white;'>Dynamic Model Predictions:</span> <span style='color: green; font-size: 20px;'>{round(dynamic_predictions[0], 4)}</span>", unsafe_allow_html=True)
            st.markdown(f"<span style='color: white;'>Demand Model Predictions:</span> <span style='color: green; font-size: 20px;'>{round(demand_predictions[0], 4)}</span>", unsafe_allow_html=True)
            eta = 'this part should be predicted'
            st.markdown(f"<span style='color: white;'>Estimated Demand:</span> <span style='color: green; font-size: 20px;'>{eta}</span>", unsafe_allow_html=True)
        else:
            st.warning("No data available for predictions. Please select your options.")
        

