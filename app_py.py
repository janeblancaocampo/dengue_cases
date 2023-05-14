import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import plotly.express as px
import matplotlib.pyplot as plt

# load the saved LSTM model
model = load_model('lstm_model.h5')

# function to make predictions using the loaded model
def predicted_case(data):
    # slice the data to only include the last months
    data = data[-60:]
    # reshape the input data to match the expected input shape of the model
    data = np.reshape(data, (data.shape[0], 1, 0))
    # make the prediction
    prediction = model.predict(data)
    # return the predicted temperature value
    return prediction[0][0]

# set up the Streamlit app
st.set_page_config(page_title="Dengue Cases in Sri Lanka Predictor")
st.title("Dengue Cases Predictor")
st.write("This app predicts the next Dengue Cases in Sri Lanka")

# load the temperature data
data = pd.read_csv('Dengue_Data (2010-2020).csv', parse_dates=['Date'])
data.set_index('Date', inplace=True)

# display the current dengue cases data
st.subheader("Current Dengue Cases in Sri Lanka")
st.write(data.tail())

# get the latest dengue cases 
latest_case = data['Value'].iloc[-1]

# get the date of the latest dengue cases
latest_date = data.index[-1]

# make predictions for the next 12 months
predicted_cases = []
for i in range(12):
    next_date = latest_date + pd.DateOffset(months=1)
    pred_case = predicted_case(data['Value'])
    predicted_cases.append(pred_case)
    latest_date = next_date
    data.loc[next_date] = pred_case

# display the predicted cases for the next 12 months
st.subheader("Next 12 Months' Dengue Cases")
st.write(data.tail(12)['Value'])

# plot the predicted cases
fig = px.line(data.tail(12), y='Value', title="Predicted Dengue Cases Graph for the next 12 Months")
st.plotly_chart(fig)

# plot the actual and predicted cases
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(data['Value'], label='Actual')
ax.plot(data['Value'].tail(12), label='Predicted')
ax.set_xlabel('Year')
ax.set_ylabel('Dengue Cases')
ax.set_title('Actual vs Predicted Dengue Cases in Sri Lanka')
ax.legend()
st.pyplot(fig)
