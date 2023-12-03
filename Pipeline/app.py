import streamlit as st
import pandas as pd
from preprocess import get_base_data, get_dynamic_data
from train import prepare_data, train_and_save_model

# Set page title
st.title("Simulation for Cab Ride Price Optimization")

# Explain the problem
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
        🙌  This section can use colors to highlight important points like <span class='text-red'>a</span> and <span class='text-red'>b</span> <br>
        <br>
    </div>
    """, 
    unsafe_allow_html=True
)

# Load data with a spinner
with st.spinner('Loading data...'):
    base_df = get_base_data()
    dynamic_df = get_dynamic_data()

st.success('Data loaded!')

# Initialize session state variables for filtered dataframes and training status
if 'filtered_base_df' not in st.session_state:
    st.session_state['filtered_base_df'] = base_df
if 'filtered_dynamic_df' not in st.session_state:
    st.session_state['filtered_dynamic_df'] = dynamic_df
if 'training_completed' not in st.session_state:
    st.session_state['training_completed'] = False

# Columns to show for user input (Filtering)
columns = ["source", "destination", "car_type", "weekday", 
           "rush_hour", "is_raining", "temp_groups"]
questions = ['Where are customers coming from?', 'Where are customers going?', 
             'What type of service?', 'Weekday or Weekend?', 'Is it rush hour?', 'Is it raining?', 
             'What is the temperature group?']

# Select Cab Type
st.header("1. Select Cab Type")
cab_type_options = ['Uber', 'Lyft']
cab_type = st.radio("Uber or Lyft?", cab_type_options, index=0 if 'cab_type' not in st.session_state else cab_type_options.index(st.session_state.cab_type))

# Define car_type options based on cab_type selection
car_type_options = {
    'Uber': ['Luxury', 'Base XL', 'Base', 'Wheel Chair Accessible', 'Luxury SUV', 'Shared'],
    'Lyft': ['Luxury SUV', 'Base', 'Shared', 'Luxury', 'Base XL']
}

# Make all select boxes as a group using st.form
st.header("2. Select Data Filters")
with st.form(key='filter_form'):
    filters = {'cab_type': cab_type}
    temp_filtered_base_df = base_df[base_df['cab_type'] == cab_type]
    temp_filtered_dynamic_df = dynamic_df[dynamic_df['cab_type'] == cab_type]
    
    for col, question in zip(columns, questions):
        if col == "car_type":
            options = car_type_options[cab_type]
        elif col in ["weekday", "rush_hour", "is_raining"]:
            options = ['Yes', 'No']
            default_index = int(temp_filtered_base_df[col].mode()[0]) if not temp_filtered_base_df[col].mode().empty else 0
        else:
            options = sorted(temp_filtered_base_df[col].unique())
            default_index = 0

        selected_option = st.selectbox(question, options, index=default_index, key=col)
        if col in ["weekday", "rush_hour", "is_raining"]:
            selected_value = 1 if selected_option == 'Yes' else 0
            temp_filtered_base_df = temp_filtered_base_df[temp_filtered_base_df[col] == selected_value]
            temp_filtered_dynamic_df = temp_filtered_dynamic_df[temp_filtered_dynamic_df[col] == selected_value]
        else:
            temp_filtered_base_df = temp_filtered_base_df[temp_filtered_base_df[col] == selected_option]
            temp_filtered_dynamic_df = temp_filtered_dynamic_df[temp_filtered_dynamic_df[col] == selected_option]

        filters[col] = selected_option

    submit_button = st.form_submit_button(label='Apply Filters')

if submit_button:
    st.session_state['filtered_base_df'] = temp_filtered_base_df
    st.session_state['filtered_dynamic_df'] = temp_filtered_dynamic_df
    st.session_state['training_completed'] = False  # Reset training status when filters are applied
    st.success('Filters applied and data filtered!')

# Model Training and Saving Section
if not st.session_state['training_completed']:
    st.header("3. Train and Save Models")
    if st.button("Train and Save Models"):
        with st.spinner('Training models...'):
            try:
                X_base, y_base, preprocessor_base = prepare_data(st.session_state['filtered_base_df'])
                train_and_save_model(X_base, y_base, preprocessor_base, 'base_model')
                
                X_dynamic, y_dynamic, preprocessor_dynamic = prepare_data(st.session_state['filtered_dynamic_df'])
                train_and_save_model(X_dynamic, y_dynamic, preprocessor_dynamic, 'dynamic_model')

                st.session_state['training_completed'] = True
                st.success("All models trained and saved successfully!")
            except Exception as e:
                st.session_state['training_completed'] = False  # Reset training status on error
                st.error(f"An error occurred: {e}")
else:
    st.success("All models trained and saved successfully!")
