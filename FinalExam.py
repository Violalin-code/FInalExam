import streamlit as st
import joblib
import pickle
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained ANN model
model = load_model('ann_model.h5')

# Load the scaler for preprocessing
scaler = joblib.load('scaler (15).pkl')

# Load feature and target columns
with open('columns.pkl', 'rb') as f:
    X_columns = pickle.load(f)
    y_columns = pickle.load(f)

def encode_input_data(input_data):
    """
    Encode input data using pandas get_dummies
    """
    # Create DataFrame from input
    input_df = pd.DataFrame([input_data])
    
    # Get material columns from X_columns
    material_columns = [col for col in X_columns if col.startswith('Material_')]
    
    # Create encoded dataframe with all X_columns, initialized to 0
    encoded_df = pd.DataFrame(0, index=[0], columns=X_columns)
    
    # Safe mapping for materials to handle unexpected or missing materials
    material_col = f"Material_{input_data['Material']}"
    
    # If the material column exists in X_columns, set it to 1, otherwise, map or skip
    if material_col in X_columns:
        encoded_df[material_col] = 1
    else:
        material_mapping = {
            'Alfafa flower': 'Alfalfa flower',  # Correct the spelling
            'Rice straw ': 'Rice straw',  # Remove trailing space
            'Pine Nut Shell ': 'Pine nut shell',  # Standardize capitalization
            'Tobacco stalk    ': 'Tobacco stalk'  # Clean extra spaces
        }
        corrected_material = material_mapping.get(input_data['Material'], input_data['Material'])
        material_col = f"Material_{corrected_material}"
        
        # Check if corrected material column exists, if so, set it, else skip
        if material_col in X_columns:
            encoded_df[material_col] = 1
    
    # Handle other categorical variables similarly
    for cat_var in ['KB', 'Purifying', 'Electrode system (2E/3E)', 'Electrolyte Kind']:
        col_name = f"{cat_var}_{input_data[cat_var]}"
        if col_name in X_columns:
            encoded_df[col_name] = 1
    
    # Add numerical columns
    numerical_columns = [col for col in X_columns if not any(col.startswith(prefix) for prefix in 
                        ['Material_', 'KB_', 'Purifying_', 'Electrode system (2E/3E)_', 'Electrolyte Kind_'])]
    
    for col in numerical_columns:
        if col in input_data:
            encoded_df[col] = input_data[col]
    
    return encoded_df

def make_prediction(input_data):
    """
    Process input data and make predictions using the model
    """
    try:
        # Encode the input data
        encoded_input = encode_input_data(input_data)
        
        # Scale the features
        scaled_features = scaler.transform(encoded_input)
        
        # Make prediction
        prediction = model.predict(scaled_features)
        
        return prediction[0][0], prediction[0][1]
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        raise e

# Streamlit interface
st.title('Capacitance and Rate Capability Prediction')

# Input form
input_data = {}

# Categorical inputs
materials = [
    'Willow', 'Orange peel', 'Loofah sponge', 'Pine nut shell', 'Gelatin',
    'Bamboo', 'Waste Palm', 'Maize', 'Wheat straw', 'Waste printing paper',
    'Sisal Fiber', 'Rice straw ', 'Pineapple leaf', 'Areca nut', 'Peanut shell',
    'Rice husk', 'Cabbage leaves', 'Cotton', 'Pine nut shells', 'Houttuynia',
    'Fromlaminaria japonica', 'Alfafa flower', 'Quinoa', 'Soybean straw',
    'Fir sawdust', 'Pine Nut Shell ', 'Eulaliopsis binata', 'Bean dregs',
    'Bagasse', 'Fresh clover stems', 'Longan shells', 'Wood sawdust', 'Hemp',
    'Leftover rice', 'Wallnut Shell', 'Corn straw', 'Kapok fiber',
    'Xanthoceras sorbifolia seed coats', 'Soybean Leaf', 'Tobacco stalk    ',
    'Rice straw', 'Cashew nut husk', 'Corncob', 'Garlic Skin', 'Raw rice brans',
    'Bambo', 'Wastetealeaves', 'Corn stalk core', 'Cattail wool',
    'Coniferous pine', 'Banana peels', 'Garlic peel', 'Bamboo fibers',
    'Pine needles', 'Pine cone', 'Celtuce Leaves', 'Tea Leaves', 'Horseweed',
    'Coconut shells'
]

input_data['Material'] = st.selectbox('Material', materials)
input_data['KB'] = st.selectbox('KB', ['PTFE', 'None'])
input_data['Purifying'] = st.selectbox('Purifying', ['none', 'water', 'diluted HCl', 'diluted HCl + water'])
input_data['Electrode system (2E/3E)'] = st.selectbox('Electrode system', ['2E', '3E'])
input_data['Electrolyte Kind'] = st.selectbox('Electrolyte Kind', ['KOH', 'H2SO4', 'NaNO3', 'TEABF4/PC'])

# Pre-carbonization temperature with proper type handling
pre_carb_options = [
    500.0, 550.0, 600.0, 400.0, 800.0, 200.0, 300.0, 'none', 850.0, 100.0, 
    180.0, 160.0, 220.0, 150.0, 700.0, 900.0, 240.0, 1000.0, 450.0, 1050.0
]
input_data['pre carbonization (°C)'] = st.selectbox('Pre carbonization (°C)', pre_carb_options)

# Numerical inputs
numerical_columns = [
    'Ratio of activation Agent', 'Time (h)', 'Carbonization (°C)',
    'Time (h).1', 'ACR (%)', 'CMR (%)', 'BR (%)', 'CM/AC', 'B/AC', 'CM/B',
    'V0.1 (cm^3/g)', 'V0.4 (cm^3/g)', 'V0.9 (Cm^3/g)',
    'Molarity of HCl (M)', 'Molarity of Electrolyte (M)', 'Potential Window (V)'
]

# Create two columns for numerical inputs
col1, col2 = st.columns(2)

with col1:
    for col in numerical_columns[:len(numerical_columns)//2]:
        input_data[col] = st.number_input(
            col,
            value=0.0,
            step=0.1,
            help=f"Enter the value for {col}"
        )

with col2:
    for col in numerical_columns[len(numerical_columns)//2:]:
        input_data[col] = st.number_input(
            col,
            value=0.0,
            step=0.1,
            help=f"Enter the value for {col}"
        )

# Make prediction when button is clicked
if st.button('Predict'):
    try:
        with st.spinner('Making prediction...'):
            specific_capacitance, rate_capability = make_prediction(input_data)
        
        # Display results
        st.success('Prediction completed!')
        
        # Display only the Specific Capacitance value
        st.metric(
            label="Specific Capacitance (F/g)",
            value=f"{specific_capacitance:.2f}"
        )
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Input data:", input_data)
