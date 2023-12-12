from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from joblib import dump
from joblib import load

def load_models():
    base_model = load('Models/base_model_pipeline.joblib')
    dynamic_model = load('Models/dynamic_model_pipeline.joblib')
    demand_model = load('Models/demand_model_pipeline.joblib')
    return base_model, dynamic_model, demand_model 

def predict_prices(new_data, base_model, dynamic_model, demand_model):
    # if 'date_time' in new_data.columns:
    #     new_data.set_index('date_time', inplace=True)
    base_predictions = base_model.predict(new_data)
    dynamic_predictions = dynamic_model.predict(new_data)
    demand_predictions = demand_model.predict(new_data)
    return base_predictions, dynamic_predictions, demand_predictions

def predict_eta(new_data, eta_model=load('Models/demand_model_eta_pipeline.joblib')):
    eta_predictions = eta_model.predict(new_data)
    return eta_predictions


