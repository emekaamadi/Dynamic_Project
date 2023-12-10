# Capstone Project Report: Dynamic Pricing Analysis Using Uber and Lyft Data

## Project Overview
**"Dynamic_Project"** delves into the nuances of dynamic pricing in Uber and Lyft services. It aims to analyze the fare differences in standard and dynamically priced rides. A web application is developed for users to visualize price variations and understand the dynamic pricing strategies of these ride-sharing companies.

## Dynamic Pricing Explored
Dynamic pricing, a strategy where prices are adjusted based on demand, is common in industries like airlines. This report examines this concept using Uber and Lyft data, integrated with weather information, to explore fare discrepancies under different conditions and pricing models.

## GitHub Repository and Documentation
The project is documented on GitHub, providing a README file for execution guidance, a requirements.txt for dependencies, and the core codebase. Note: Due to access restrictions, our data is sourced from Kaggle, not directly obtained via Uber and Lyft APIs.

## Code and Data Management

#### Cleaning Engineering
- **Data Preparation**: Using Pandas, we imported and preprocessed cab_rides.csv and weather.csv datasets, involving data cleaning and feature engineering.
- **Weather Categorization**: Rainfall data was binary encoded, and temperatures were grouped for pattern analysis.
- **Vehicle Categorization and Time Analysis**: Vehicles were categorized by type, and additional columns for weekdays and rush hours were added.
- **Data Merging**: Merged trip data with corresponding weather information for a comprehensive dataset.

#### Dynamic Modeling
- **Data Analysis**: Employed statistical methods to analyze fare distributions across different vehicle types and locations.
- **Visualization**: Used Seaborn and Matplotlib for visual representation of fare distributions.
- **Demand Calculation**: Developed a method to calculate demand based on price metrics, crucial for pricing model development.

#### Thompson Sampling Analysis
- **Model Building**: Built a probabilistic model with PyMC3 to estimate demand parameters.
- **Data Integration**: Integrated estimated demand parameters with the original dataset for enriched analysis.

#### User Modeling
- **Model Training**: Used various regression models from Scikit-learn for price prediction.
- **Data Preprocessing and Validation**: Implemented data preprocessing techniques and evaluated model performance using MSE and R^2 score.

### Pipeline
- **Data Processing**: Developed a pipeline for data handling and machine learning to predict taxi fares dynamically.
- **Model Implementation**: Chose a Decision Tree Regression model, with results saved for future application.


## Visualizations

### Cleaning Engineering
- **Statistical Exploration**: Utilized gamma distribution with Scipy and Matplotlib, and performed descriptive statistical analysis on demand estimates and trip pricing.

### Dynamic Modeling
- **Data Processing Pipeline**: Developed a pipeline for dynamic taxi fare prediction using Pandas, including data cleaning, feature extraction, and model training with a Decision Tree Regression model.

### Thompson Sampling Analysis
- **Bayesian Modeling**: Detailed demand estimation using PyMC3, processing data with Theano tensors, and defining prior distributions for key parameters (eta, a, b).
- **Parameter Estimation**: Focused on sampling from the model's posterior distribution using MCMC, presenting estimates for parameters affecting demand.
- **Visualization and Analysis**: Showcased demand estimation functions and provided visual representations of demand variations with price changes.

### User Modeling
- **Machine Learning Models**: Discussed data preprocessing, training, and validation of various models, with a focus on Decision Tree Regression.
- **Model Optimization**: Employed grid search and cross-validation for optimizing model parameters, enhancing prediction accuracy and feature impact analysis.
