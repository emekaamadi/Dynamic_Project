# MADS 699 team 15 Capstone Project: Dynamic Pricing Analysis Using Uber and Lyft Data

## Project Overview and Details

### 1. Overview
The "Dynamic_Project" investigates dynamic pricing in Uber and Lyft, focusing on how fares differ between standard and dynamically priced rides. It encompasses data analysis, model development, and an interactive web application for visualization and comparison of pricing strategies.

### Data Access Statement
Accessing the APIs of Uber and Lyft requires authorization, which we were unable to obtain. Instead, our data was sourced from [Kaggle](https://www.kaggle.com/datasets/ravi72munde/uber-lyft-cab-prices), where datasets were originally gathered through the Uber and Lyft APIs. This data represents what the price and surge data of a ride would be if a ride was taken at that time. It's important to note that Uber and Lyft actual ride data is not publicly available.
<br> 

### 2. Details

### Data Analysis
- **Data Sources**: Utilizes Uber and Lyft data along with weather information.
- **Insightful Analysis**: Investigates conditions affecting fare differences between base and dynamic pricing.

### Model Development
- **Demand Estimation Model**: Inspired by economic demand functions, predicts prices using estimated parameters.
- **Multiple Models**: Develops base, dynamic, and demand estimation models for comprehensive analysis.

### Web Application
- **Interactive Tool**: Built using Streamlit, it enables users to explore and compare different pricing models.
- **Visualizations**: Offers insights into demand curves and revenue estimations based on the models' predictions.

### User Experience
- **Customizable Inputs**: Users can specify conditions like origin, destination, and service type.
- **Real-Time Analysis**: Provides instant visual feedback and price predictions based on user inputs.

### Conclusion
The project exemplifies the use of advanced data analysis and modeling techniques to understand and visualize the complex dynamics of ride-sharing pricing strategies.

<br>
<br>
<br>

## Pipeline Detailed Description

### Preprocessing: `preprocess.py`
- **Data Loading and Preprocessing**: Responsible for loading files and executing all data preprocessing functions.
- **Data Transformation**: Transforms the data into a suitable format for further processing and analysis.

### Training Models: `train.py`
- **Model Creation**: Generates three types of models: base, dynamic, and demand.
- **Model Saving**: Saves the trained models for future prediction use.

### Demand Estimation: `demand_estimation.py`
- **Economic Demand Function**: Utilizes the formula \$Q = a \cdot P^{-\eta} + b\$ for demand estimation.
- **Parameter Estimation**: Employs Markov Chain Monte Carlo (MCMC) to estimate eta, a, and b parameters, usually taking about 10 hours.
- **Pre-generated Data**: Utilizes a pre-generated CSV file to bypass the time-intensive MCMC process.

### Predictions: `predict.py`
- **Model Utilization**: Uses the trained models from `train.py` for making predictions.

### Application Interface: `app.py`
- **Streamlit Application**: Allows corporate users to input various conditions like origin, destination, and service type.
- **Visualization**: Provides two visualizations: 
  1. Estimated demand curve with predicted prices from each model plotted.
  2. Bar chart representation of the calculated revenue based on these estimates.

### Execution: `run.py`
- **Process Execution**: Orchestrates the entire pipeline. If the demand estimation CSV file does not exist, it initiates its creation, which can take up to 10 hours. If the file already exists, it uses the pre-generated data, skipping the time-intensive demand estimation process.
- **Note**: Creating the CSV file for demand estimation is a lengthy process, please be aware of this time commitment.

<br>
<br>
<br>

## Environment Setup and Execution Guide

Before running `run.py`, it's essential to set up the project environment correctly. This guide provides step-by-step instructions for setting up the environment using either Conda or venv and installing necessary packages.

### Setting Up Using Conda

1. **Create a New Conda Environment**:
   - Open your terminal.
   - Create a new Conda environment by running:
     ```bash
     conda create --name myenv python=3.8
     ```
   - Replace `myenv` with your desired environment name.

2. **Activate the Conda Environment**:
   - Activate the newly created environment:
     ```bash
     conda activate myenv
     ```

### Setting Up Using venv

1. **Create a New Virtual Environment**:
   - Open your terminal.
   - Navigate to your project directory.
   - Create a new virtual environment by running:
     ```bash
     python -m venv myenv
     ```
   - Replace `myenv` with your desired environment name.

2. **Activate the Virtual Environment**:
   - Activate the environment:
     - On Windows:
       ```bash
       .\myenv\Scripts\activate
       ```
     - On macOS/Linux:
       ```bash
       source myenv/bin/activate
       ```

### Installing Dependencies

1. **Install Required Packages**:
   - Ensure your environment is activated.
   - Install all required packages using:
     ```bash
     pip install -r requirements.txt
     ```

### Running the Application

1. **Execute `run.py`**:
   - With the environment set up and dependencies installed, navigate to the Pipeline folder and run the application:
     ```bash
     python run.py
     ```

Following these steps will ensure that your environment is correctly configured and all dependencies are installed, allowing `run.py` to run smoothly.

<br>
<br>
<br>

## Generating and Saving Plots

This project includes a Python file called `util.py` for generating and saving plots. Plots are saved in the Visuals folder.
Some of the plots are listed below.
Ensure you have the required dependencies (`numpy`, `matplotlib`, `scipy`) installed to generate and view these plots. Call the functions with your desired parameters.
