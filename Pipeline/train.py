from preprocess import get_base_data, get_dynamic_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from joblib import dump

def prepare_data(data):
    X = data.drop('price', axis=1)
    y = data['price']

    # Preprocessing for numerical features
    numerical_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]
    numerical_transformer = StandardScaler()

    # Preprocessing for categorical features
    categorical_cols = [cname for cname in X.columns if X[cname].dtype == 'object']
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Bundle preprocessing for numerical and categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)])

    return X, y, preprocessor

def train_and_save_model(X, y, preprocessor, model_name):
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', DecisionTreeRegressor(random_state=0))])

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
    pipeline.fit(X_train, y_train)

    # Save the model
    dump(pipeline, f'Models/{model_name}_pipeline.joblib')

# if __name__ == "__main__":
#     # Process and save model for base data
#     base_data = get_base_data()
#     X, y, preprocessor = prepare_data(base_data)
#     train_and_save_model(X, y, preprocessor, 'base_model')
#     print("Saved Base Model")

#     # Process and save model for dynamic data
#     dynamic_data = get_dynamic_data()
#     X, y, preprocessor = prepare_data(dynamic_data)
#     train_and_save_model(X, y, preprocessor, 'dynamic_model')
#     print("Saved Dynamic Model")

    # Process and save model for demand data 
    # either import saved file or create funtion to get data 
    #X, y, preprocessor = prepare_data(demand_data)
    #train_and_save_model(X, y, preprocessor, 'demand_model')
    #print("Saved Demand Model")