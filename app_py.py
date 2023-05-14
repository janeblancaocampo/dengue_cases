import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Load the data
data = pd.read_csv('Dengue_Data (2010-2020).csv', index_col = 'Date', parse_dates = True)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data.values)

# Load the model
model = tf.keras.models.load_model('lstm_model.h5')

# Create a function to make predictions
def make_predictions(data, model, n_predictions):
    # Reshape the input data
    X = data[-1:,1:]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    # Make predictions for the next n predictions
    predictions = []
    for i in range(n_predictions):
        y_hat = model.predict(X)
        predictions.append(y_hat[0][0])
        X = np.append(X[:,:,1:], [[y_hat[0]]], axis=2)
    return predictions

# Set the number of predictions to make
n_predictions = 12

# Make predictions
predictions = make_predictions(scaled_data, model, n_predictions)
predictions = scaler.inverse_transform(np.array(predictions).reshape(-1,1)).flatten()

# Show the current dengue cases and predicted dengue cases for the next 12 months
st.write("## Dengue Cases")
st.write("Current Cases: ", int(data['Value'].iloc[-1]))
st.write("Predicted Cases for the Next 12 Months: ", [int(p) for p in predictions])

# Show a graph of the current and predicted dengue cases
st.write("## Dengue Cases Graph")
df = pd.DataFrame({'Date': data.index, 'Cases': data['Value']})
df = df.append({'Date': pd.date_range(start=data.index[-1], periods=2, freq='MS')[1], 'Cases': predictions[0]}, ignore_index=True)
for i in range(1, len(predictions)):
    df = df.append({'Date': pd.date_range(start=df['Date'].iloc[-1], periods=2, freq='MS')[1], 'Cases': predictions[i]}, ignore_index=True)
df.set_index('Date', inplace=True)
st.line_chart(df)
