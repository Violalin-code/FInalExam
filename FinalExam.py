import streamlit as st
import joblib
import pickle
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained ANN model
model = load_model('ann_model.h5')

# Load the scaler for preprocessing
scaler = joblib.load('scaler (15).pkl')

# Load feature and target columns (for reference and validation)
with open('columns.pkl', 'rb') as f:
    X_columns = pickle.load(f)
    y_columns = pickle.load(f)

def make_prediction(input_data):
    # Debugging line to check what columns are being passed
    print("Input data:", input_data)  # Print the input data
    print("X_columns:", X_columns)   # Print the expected feature columns
    
    input_features = np.array([input_data[column] for column in X_columns]).reshape(1, -1)
    
    input_scaled = scaler.transform(input_features)  # Apply scaling
    prediction = model.predict(input_scaled)  # Make prediction
    return prediction[0][0], prediction[0][1]  # Return both values

# Streamlit interface
st.title('Capacitance and Rate Capability Prediction')

# Input form for features (you can modify the form to match all of your feature columns)
input_data = {}
input_data['Material'] = st.selectbox('Material', [
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
])

input_data['Doping Agent'] = st.selectbox('Doping Agent', ['none', 'melamine', '(NH4)2B4O7$4H2O', 'urea'])
input_data['Activation Agent'] = st.selectbox('Activation Agent', ['KOH', 'H2SO4', 'ZnCl2', 'none', 'KCl'])

input_data['pre carbonization (°C)'] = st.selectbox('Pre carbonization (°C)', [
    500.0, 550.0, 600.0, 400.0, 800.0, 200.0, 300.0, 'none', 850.0, 100.0, 180.0, 160.0,
    220.0, 150.0, 700.0, 900.0, 240.0, 1000.0, 450.0, 1050.0
])

input_data['KB'] = st.selectbox('KB', ['PTFE', 'PVDF', 'none'])
input_data['Purifying'] = st.selectbox('Purifying', ['diluted HCl + water', 'water', 'none', 'diluted HCl'])
input_data['Electrode system (2E/3E)'] = st.selectbox('Electrode system (2E/3E)', ['3E', '2E'])
input_data['Electrolyte Kind'] = st.selectbox('Electrolyte Kind', ['KOH', 'H2SO4', 'TEABF4/PC', 'NaNO3'])

# Numerical input fields
input_data['Ratio of activation Agent'] = st.number_input('Ratio of activation Agent', min_value=0.0)
input_data['Time (h)'] = st.number_input('Time (h)', min_value=0)
input_data['Time (h).1'] = st.number_input('Time (h).1 (secondary time)', min_value=0)  # Secondary time field
input_data['Carbonization (°C)'] = st.number_input('Carbonization (°C)', min_value=0)
input_data['ACR (%)'] = st.number_input('ACR (%)', min_value=0.0)
input_data['CMR (%)'] = st.number_input('CMR (%)', min_value=0.0)
input_data['BR (%)'] = st.number_input('BR (%)', min_value=0.0)
input_data['CM/AC'] = st.number_input('CM/AC', min_value=0.0)
input_data['B/AC'] = st.number_input('B/AC', min_value=0.0)
input_data['CM/B'] = st.number_input('CM/B', min_value=0.0)
input_data['V0.1 (cm^3/g)'] = st.number_input('V0.1 (cm^3/g)', min_value=0.0)
input_data['V0.4 (cm^3/g)'] = st.number_input('V0.4 (cm^3/g)', min_value=0.0)
input_data['V0.9 (Cm^3/g)'] = st.number_input('V0.9 (Cm^3/g)', min_value=0.0)
input_data['Molarity of HCl (M)'] = st.number_input('Molarity of HCl (M)', min_value=0.0)
input_data['Molarity of Electrolyte (M)'] = st.number_input('Molarity of Electrolyte (M)', min_value=0.0)
input_data['Potential Window (V)'] = st.number_input('Potential Window (V)', min_value=0.0)

# When the user presses the button, make the prediction
if st.button('Predict'):
    # Convert 'Yes'/'No' to binary values for 'Purifying' and 'Electrode system (2E/3E)'
    input_data['Purifying'] = 1 if input_data['Purifying'] == 'diluted HCl + water' else 0
    input_data['Electrode system (2E/3E)'] = 1 if input_data['Electrode system (2E/3E)'] == '2E' else 0

    # Make the prediction
    specific_capacitance, rate_capability = make_prediction(input_data)

    # Display the results
    st.subheader('Prediction Results:')
    st.write(f"Specific Capacitance (F/g): {specific_capacitance:.2f}")
    st.write(f"Rate Capability (F/g): {rate_capability:.2f}")