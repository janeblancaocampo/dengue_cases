import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import plotly.express as px
import matplotlib.pyplot as plt

# load the saved LSTM model
model = load_model('lstm_model.h5')

# function to make predictions using the loaded model
def predict_case(data):
    # slice the data to only include the last 60 days
    data = data[-60:]
    # reshape the input data to match the expected input shape of the model
    data = np.reshape(data, (1, -1, data.shape[1]))
    prediction = model.predict(data)
    # return the predicted dengue case value
    return prediction[0][0]

# set up the Streamlit app
st.set_page_config(page_title="Monthly Predicted Dengue Cases in Sri Lanka")
st.title("Monthly Predicted Dengue Cases in Sri Lanka")
st.write("This app predicts the nonthly dengue cases in Sri Lanka based on historical data.")

# load the data
data = pd.read_csv('Dengue_Data (2010-2020).csv', parse_dates=['Date'])
data.set_index('Date', inplace=True)

# display the current temperature data
st.subheader("Current Dengue Cases")
st.write(data.tail())

# get the latest dengue cases
latest_case = data['Value'].iloc[-1]

# get the date of the latest temperature value
latest_date = data.index[-1]

# make predictions for the next 12 months
predicted_cases = []
for i in range(30):
    next_date = latest_date + pd.Timedelta(days=1)
    pred_case = predict_case(np.array(data['Value']))
    predicated_cases.append(pred_case)
    latest_date = next_date
    data.loc[next_date] = pred_case

# display the predicted dengue cases for the next 12 months
st.subheader("Next 12 months of Dengue Cases in Sri Lanka")
st.write(data.tail(12)['Value'])

# plot the predicted dengue cases using Plotly Express
fig = px.line(data.tail(12), y='Value', title="Predict Dengue Cases for the next 12 months")
st.plotly_chart(fig)

# plot the predicted dengue values
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(data['Value'], label='Actual')
ax.plot(data['Value'].tail(30), label='Predicted')
ax.set_xlabel('Year')
ax.set_ylabel('Value')
ax.set_title('Actual vs Predicted Monthly Predicted Dengue Cases')
ax.legend()
st.pyplot(fig)
