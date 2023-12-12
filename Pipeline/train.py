from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from joblib import dump
from joblib import load


def prepare_data(data):
    if "date_time" in data.columns:
        X = data.drop(['date_time'], axis=1)
    X = data.drop(['price'], axis=1)
    # add date_time to previous if you added it to the get functions
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


def prepare_data_eta(data):
    # if "date_time" in data.columns:
    #     X = data.drop(['date_time'], axis=1)
    X = data.drop('estimated_eta', axis=1)
    y = data['estimated_eta']

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


def train_and_save_model_for_eta(X, y, preprocessor, model_name):
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', DecisionTreeRegressor(random_state=0))])

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
    pipeline.fit(X_train, y_train)

    # Save the model
    dump(pipeline, f'Models/{model_name}_pipeline.joblib')
