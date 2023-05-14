import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model

# Load the pre-trained LSTM model
model = load_model('lstm_model.h5')

# Load the data
data = pd.read_csv('Dengue_Data (2010-2020).csv', index_col='Date', parse_dates=True)

# Preprocess the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data.values)

# Split the data into input and output variables
X = scaled_data[:,1:]
y = scaled_data[:,0]

# Reshape input data into (samples, time steps, features)
X = X.reshape(X.shape[0], 1, X.shape[1])

# Make predictions
predictions = model.predict(X)

# Inverse transform the predictions and actual values
predictions = scaler.inverse_transform(predictions)
y = scaler.inverse_transform(y.reshape(-1, 1))

# Get the current dengue cases and predicted dengue cases for the next 12 months
current_cases = int(y[-1][0])
predicted_cases = int(predictions[-1][0])
for i in range(1, 13):
    predicted_cases = int(model.predict(predictions[-1].reshape(1, 1, 1))[0][0])
    predictions = np.append(predictions, predicted_cases)
predicted_cases = int(predictions[-1])

# Create a DataFrame for the current and predicted dengue cases
dates = pd.date_range(start=data.index[-1], periods=13, freq='MS')[1:]
predicted_cases_df = pd.DataFrame({'Date': dates, 'Dengue Cases': predictions[-12:]})

# Create a line plot of the current and predicted dengue cases
st.title('Dengue Cases Forecast')
st.write('Current Dengue Cases:', current_cases)
st.write('Predicted Dengue Cases (Next 12 Months):', predicted_cases)
st.line_chart(predicted_cases_df.set_index('Date'))
