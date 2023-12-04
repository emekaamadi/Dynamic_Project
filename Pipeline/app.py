import streamlit as st
import pandas as pd
from preprocess import get_base_data, get_dynamic_data
from predict import load_models, predict_prices

########## Set Title ##########
st.title("Simulation for Cab Ride Price Optimization")

########## Explain the problem ##########
st.header("0. Problem Explanation")
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

########## Load CSV file ##########
st.header("1. Upload CSV File")
uploaded_file = st.file_uploader("Choose a file", type="csv")
if uploaded_file is not None:
    uploaded_data = pd.read_csv(uploaded_file)
    base_df = get_base_data(uploaded_data)
    dynamic_df = get_dynamic_data(uploaded_data)
    st.success('File uploaded and data processed successfully!')

    ########## Set Filtering Options ##########
    st.header("2. Select Filters")
    if uploaded_file is not None:
        # Select Cab Type
        cab_type_options = ['Uber', 'Lyft']
        cab_type = st.radio("Uber or Lyft?", cab_type_options, key="cab_type")

        # Initialize filtered dataframes
        filtered_base_df = base_df.copy()
        filtered_dynamic_df = dynamic_df.copy()

        # Apply Filters for Cab Type
        if cab_type:
            filtered_base_df = filtered_base_df[filtered_base_df['cab_type'] == cab_type]
            filtered_dynamic_df = filtered_dynamic_df[filtered_dynamic_df['cab_type'] == cab_type]

        # Define car_type options based on cab_type selection
        car_type_options = {
            'Uber': ['Luxury', 'Base XL', 'Base', 'Wheel Chair Accessible', 'Luxury SUV', 'Shared'],
            'Lyft': ['Luxury SUV', 'Base', 'Shared', 'Luxury', 'Base XL']
        }

        # Filtering form
        with st.form(key='filter_form'):
            # Customize Options
            columns = ["source", "destination", "car_type", "weekday", "rush_hour", "is_raining", "temp_groups"]
            questions = ['Where are customers coming from?', 'Where are customers going?', 'What type of service?', 'Weekday or Weekend?', 'Is it rush hour?', 'Is it raining?', 'What is the temperature group?']

            for col, question in zip(columns, questions):
                if col == "car_type":
                    options = car_type_options.get(cab_type, sorted(filtered_base_df[col].unique()))
                elif col == "weekday":
                    options = ['Weekday', 'Weekend']
                elif col in ["rush_hour", "is_raining"]:
                    options = ['Yes', 'No']
                elif col == "temp_groups":
                    options = ['20 degrees or less', '20-30 degrees', '30-40 degrees', '40-50 degrees', '50 or more']
                else:
                    options = sorted(filtered_base_df[col].unique())

                selected_option = st.selectbox(question, options, key=col)

                # Dynamically update the dataframe based on selection
                if options and selected_option in filtered_base_df[col].values:
                    filtered_base_df = filtered_base_df[filtered_base_df[col] == selected_option]
                    filtered_dynamic_df = filtered_dynamic_df[filtered_dynamic_df[col] == selected_option]

            submit_button = st.form_submit_button(label='Apply Filters')

        if submit_button:
            # Check if filtered data is empty
            if filtered_base_df.empty or filtered_dynamic_df.empty:
                st.warning("No data available with the selected filters. Please adjust your filters.")
            else:
                st.session_state['filtered_base_df'] = filtered_base_df
                st.session_state['filtered_dynamic_df'] = filtered_dynamic_df
                st.success('Filters applied and data available!')


    ########## Prediction ##########
    if 'filtered_base_df' in st.session_state and 'filtered_dynamic_df' in st.session_state:
        st.header("3. Model Predictions")
        if st.button("Predict prices"):
            # 데이터가 존재하는지 확인
            if not st.session_state['filtered_base_df'].empty:
                # Load models
                base_model, dynamic_model = load_models()

                # Conduct predictions
                base_predictions, dynamic_predictions = predict_prices(st.session_state['filtered_base_df'], base_model, dynamic_model)

                # Print the results
                # st.write("Base Model Predictions:", base_predictions[0])
                # st.write("Dynamic Model Predictions:", dynamic_predictions[0])
                st.markdown(f"<span style='color: white;'>Base Model Predictions:</span> <span style='color: green; font-size: 20px;'>{round(base_predictions[0], 4)}</span>", unsafe_allow_html=True)
                st.markdown(f"<span style='color: white;'>Dynamic Model Predictions:</span> <span style='color: green; font-size: 20px;'>{round(dynamic_predictions[0], 4)}</span>", unsafe_allow_html=True)

            else:
                st.warning("No data available for predictions. Please adjust your filters.")

