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

new_data = data[data.City == 'Colombo']
new_data.drop('City', axis = 1, inplace = True)

# Preprocess the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(new_data.values)


# Split the data into input and output variables
X_test = scaled_data[-12:, 1:]
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# Make predictions for the next 12 months
y_pred = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred)

# Create a DataFrame of the predicted values with the dates as the index
dates = pd.date_range(start=data.index[-1], periods=12, freq='MS')
pred_df = pd.DataFrame(y_pred.round(0), index=dates, columns=['Predicted Dengue Cases'])


# set up the Streamlit app
st.set_page_config(page_title="Monthly Dengue Cases Prediction", page_icon="🦟")
st.title("Monthly Dengue Cases Prediction")
st.write("This app predicts the next 12 months average dengue cases in Colomba, Sri Lanka based on historical data.")

# Getting the data with the highest cases 
highest = new_data[new_data.Value == max(new_data.Value)]
st.subheader("Highest Dengue Cases (2010 - 2020)")
highest

# Getting the data with the lowest cases 
# Getting the data with the highest cases 
lowest = new_data[new_data.Value == min(new_data.Value)]
st.subheader("Lowest Dengue Cases (2010 - 2020)")
lowest

# Show the predicted Dengue cases for the next 12 months
st.write('The predicted number of Dengue cases for the next 12 months are:')
st.write(pred_df)

# Plot the predicted Dengue cases for the next 12 months
# generate a list of y-axis tick values
y_ticks = list(range(1000, 9000, 1000))

# create the plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data['Value'], name='Actual'))
fig.add_trace(go.Scatter(x=pred_df.index, y=pred_df['Predicted Dengue Cases'], name='Predicted'))
fig.update_layout(title='Dengue Cases Prediction', xaxis_title='Date', yaxis_title='Dengue Cases', 
                  yaxis=dict(tickvals=y_ticks))
st.plotly_chart(fig)


