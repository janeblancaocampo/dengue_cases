import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import plotly.graph_objs as go

# Load the LSTM model
model = load_model('lstm_model.h5')

# Load the Dengue dataset
data = pd.read_csv('Dengue_Data (2010-2020).csv', index_col='Date', parse_dates=True)

# Preprocess the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data.values)

# Split the data into input and output variables
X_test = scaled_data[-12:, 1:]
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# Make predictions for the next 12 months
y_pred = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred)

# Create a DataFrame of the predicted values with the dates as the index
dates = pd.date_range(start=data.index[-1], periods=12, freq='MS')
pred_df = pd.DataFrame(y_pred, index=dates, columns=['Predicted Dengue Cases'])

# Get the last known value of the Dengue cases
last_known_value = data['Value'][-1]

# Create a Streamlit app
st.title('Dengue Cases Prediction')

# Show the last known value of the Dengue cases
st.write('The current number of Dengue cases is:', last_known_value)

# Show the predicted Dengue cases for the next 12 months
st.write('The predicted number of Dengue cases for the next 12 months are:')
st.write(pred_df)

# Plot the predicted Dengue cases for the next 12 months
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data['Value'], name='Actual'))
fig.add_trace(go.Scatter(x=pred_df.index, y=pred_df['Predicted Dengue Cases'], name='Predicted'))
fig.update_layout(title='Dengue Cases Prediction', xaxis_title='Date', yaxis_title='Dengue Cases')
st.plotly_chart(fig)
