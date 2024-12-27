# Biomass-based Activated Carbon (BAC) Capacitance and Rate Capability Prediction ♻️🌾🔥🌱🪵

## Overview

This web app allows users to predict the specific capacitance and rate capability of BAC electrodes based on various input parameters. The ANN model used in this project is pre-trained, and the app facilitates an easy-to-use interface where users can provide data related to material properties, activation conditions, and more. This project uses a pre-trained Artificial Neural Network (ANN) model to predict the specific capacitance and rate capability of biomass-based activated carbon (BAC) electrodes. The system is built using **Streamlit**, a Python web framework that allows users to input various parameters related to BAC synthesis and predict its electrochemical performance.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Input Parameters](#input-parameters)
- [Model Details](#model-details)
- [License](#license)

## ML Model
The machine learning model used in this project is a Pre-trained Artificial Neural Network (ANN) that predicts the specific capacitance and rate capability of biomass-based activated carbon (BAC) electrodes. The ANN is trained using a dataset that includes various input parameters such as material types, process conditions, and chemical treatments, which influence the properties of the BAC electrodes.

## Features
- **Material Selection:** Choose from a variety of materials used in BAC synthesis.
- **Process Parameters:** Input parameters related to pre-carbonization temperature, time, and other features.
- **Electrode System Selection:** Choose between 2E or 3E electrode systems.
- **Real-time Prediction:** Once the parameters are input, the app provides an instant prediction of the specific capacitance and rate capability.

## Model Files
The following files are essential for running the model in this app:
- ann_model.h5: This file contains the pre-trained ANN model. It is loaded using TensorFlow's load_model function.
- scaler (15).pkl: This file contains a saved StandardScaler used to normalize the input data before feeding it into the model. It ensures that the features are standardized and scaled properly for the model to make accurate predictions.
- columns.pkl: This file contains two objects saved using pickle:
- X_columns: List of all feature columns used by the model.
- y_columns: List of target columns. These are used for mapping the predictions back to the actual names of the targets.

## How to Run
Follow these steps to set up and run the Streamlit app for making predictions using the pre-trained machine learning model.
### 1. Install Required Libraries
- Before running the app, make sure you have all the required Python libraries installed. You can install them by creating a virtual environment and running the following commands:

### 2. Prepare Model Files
- Make sure that the following files are in the same directory as your app script (app.py):
- ann_model.h5: The pre-trained ANN model.
- scaler (15).pkl: The scaler used to standardize input data before making predictions.
- columns.pkl: Contains the feature columns (X_columns) and target columns (y_columns).
- These files are essential for loading the trained model and applying the necessary preprocessing steps.

### 3. Run the App
URL:https://huggingface.co/spaces/vjl004/Test3
