from joblib import load

# Load the pipeline
base_loaded_pipeline = load('base_model_pipeline.joblib')
dynamic_loaded_pipeline = load('dynamic_model_pipeline.joblib')
# Get the data inputed by user 
# Predict with new data
# new_data should be a DataFrame with the same features as the training data
new_data = ...  # your code to retrieve new data
base_predictions = base_loaded_pipeline.predict(new_data)
dynamic_predictions = base_loaded_pipeline.predict(new_data)